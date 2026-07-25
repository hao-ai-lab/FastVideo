# SPDX-License-Identifier: Apache-2.0
"""Validation loss and audio sampling for MMAudio feature training."""

from __future__ import annotations

import contextlib
import math
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

from fastvideo.distributed import get_local_torch_device, get_world_group
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


def _global_inference_indices(
    total_samples: int,
    world_size: int,
    rank: int,
) -> list[int | None]:
    """Assign a fixed global sample budget with equal calls on every rank."""
    if total_samples <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed MMAudio inference assignment")
    calls_per_rank = math.ceil(total_samples / world_size)
    return [
        global_index if global_index < total_samples else None for local_index in range(calls_per_rank)
        for global_index in (local_index * world_size + rank, )
    ]


def _safe_sample_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "sample"


class MMAudioValidationCallback(Callback):
    """Evaluate cached val features and periodically run native V2A inference.

    Validation loss follows the official MMAudio ``val_fn``: sample a VAE
    posterior latent, a logit-normal flow time, prior noise, and independent
    video/text CFG masks. The RNG is reset for every pass, making values at
    different training steps directly comparable.

    Optional inference reuses the live FSDP transformer in
    :class:`MMAudioPipeline`, while frozen VAE/vocoder weights are loaded from
    ``inference_model_path``. Precomputed CLIP, Synchformer, and text features
    go directly into the pipeline, so validation does not decode source video
    or run feature encoders again.
    """

    def __init__(
        self,
        *,
        data_path: str,
        every_steps: int = 5000,
        max_batches: int = 0,
        batch_size: int = 8,
        num_data_workers: int = 2,
        run_at_start: bool = False,
        use_ema: bool = False,
        inference_every_steps: int = 20000,
        inference_model_path: str = "",
        inference_num_samples: int = 16,
        inference_num_steps: int = 25,
        inference_guidance_scale: float = 4.5,
        inference_seed: int = 14159265,
        inference_save_video: bool = True,
        inference_log_to_tracker: bool = True,
        output_dir: str | None = None,
    ) -> None:
        self.data_path = str(data_path)
        self.every_steps = int(every_steps)
        self.max_batches = max(0, int(max_batches))
        self.batch_size = max(1, int(batch_size))
        self.num_data_workers = max(0, int(num_data_workers))
        self.run_at_start = bool(run_at_start)
        self.use_ema = bool(use_ema)
        self.inference_every_steps = max(0, int(inference_every_steps))
        self.inference_model_path = str(inference_model_path)
        self.inference_num_samples = max(1, int(inference_num_samples))
        self.inference_num_steps = max(1, int(inference_num_steps))
        self.inference_guidance_scale = float(inference_guidance_scale)
        self.inference_seed = int(inference_seed)
        self.inference_save_video = bool(inference_save_video)
        self.inference_log_to_tracker = bool(inference_log_to_tracker)
        self.output_dir = output_dir

        self._dataloader: Any = None
        self._pipeline: Any = None
        self._rank = 0

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        del iteration
        if not self.data_path:
            raise ValueError("MMAudio validation requires callbacks.validation.data_path")
        self._rank = get_world_group().rank
        transformer = method.student.transformer
        from fastvideo.dataset.mmaudio_feature_dataset import (
            build_mmaudio_feature_dataloader, )

        feature_shapes = {
            "latent_seq_len": int(transformer.latent_seq_len),
            "latent_dim": int(transformer.latent_dim),
            "clip_seq_len": int(transformer.clip_seq_len),
            "clip_dim": int(transformer.config.arch_config.clip_dim),
            "sync_seq_len": int(transformer.sync_seq_len),
            "sync_dim": int(transformer.config.arch_config.sync_dim),
            "text_seq_len": int(transformer.config.arch_config.text_seq_len),
            "text_dim": int(transformer.config.arch_config.text_dim),
        }
        self._dataloader = build_mmaudio_feature_dataloader(
            self.data_path,
            batch_size=self.batch_size,
            num_data_workers=self.num_data_workers,
            seed=int(self.training_config.data.seed),
            pin_memory=self.training_config.distributed.pin_cpu_memory,
            feature_shapes=feature_shapes,
            include_metadata=True,
        )
        logger.info("Initialized MMAudio validation features from %s", self.data_path)

    def on_validation_begin(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        if iteration == 0 and not self.run_at_start:
            return
        run_loss = self.every_steps > 0 and iteration % self.every_steps == 0
        run_inference = (self.inference_every_steps > 0 and bool(self.inference_model_path)
                         and iteration % self.inference_every_steps == 0)
        if not run_loss and not run_inference:
            return

        transformer = method.student.transformer
        was_training = transformer.training
        transformer.eval()
        try:
            ema_context = self._ema_context(transformer) if self.use_ema else contextlib.nullcontext(transformer)
            with ema_context:
                validation_batch = None
                if run_loss:
                    validation_batch = self._run_loss_validation(method, iteration)
                if run_inference:
                    if validation_batch is None:
                        validation_batch = next(iter(self._dataloader))
                    self._run_inference(method, validation_batch, iteration)
        finally:
            if was_training:
                transformer.train()

    def _ema_context(self, transformer: torch.nn.Module) -> Any:
        callback_dict = getattr(self, "_callback_dict", None)
        if callback_dict is not None:
            for callback in callback_dict._callbacks.values():
                if callback is self:
                    continue
                context = getattr(callback, "ema_context", None)
                if callable(context):
                    return context(transformer)
        logger.warning("MMAudio validation requested EMA weights but no EMA callback was found")
        return contextlib.nullcontext(transformer)

    @torch.inference_mode()
    def _run_loss_validation(
        self,
        method: TrainingMethod,
        iteration: int,
    ) -> dict[str, Any] | None:
        if self._dataloader is None:
            raise RuntimeError("MMAudio validation dataloader is not initialized")
        device = get_local_torch_device()
        generator = torch.Generator(device=device).manual_seed(int(self.training_config.data.seed) + self._rank, )
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        sample_count = torch.zeros((), device=device, dtype=torch.float64)
        first_batch: dict[str, Any] | None = None

        for batch_index, raw_batch in enumerate(self._dataloader):
            if self.max_batches and batch_index >= self.max_batches:
                break
            if first_batch is None:
                first_batch = raw_batch
            training_batch = method.student.prepare_batch(
                raw_batch,
                generator=generator,
                latents_source="data",
            )
            noisy_latents = training_batch.noisy_model_input
            timesteps = training_batch.timesteps
            target = training_batch.training_target
            if not all(isinstance(value, torch.Tensor) for value in (noisy_latents, timesteps, target)):
                raise RuntimeError("MMAudio validation batch is incomplete")
            assert isinstance(noisy_latents, torch.Tensor)
            assert isinstance(timesteps, torch.Tensor)
            assert isinstance(target, torch.Tensor)
            prediction = method.student.predict_noise(
                noisy_latents,
                timesteps,
                training_batch,
                conditional=True,
                attn_kind="dense",
            )
            batch_size = int(noisy_latents.shape[0])
            loss_sum += F.mse_loss(
                prediction.float(),
                target.float(),
                reduction="mean",
            ).double() * batch_size
            sample_count += batch_size

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_sum)
            torch.distributed.all_reduce(sample_count)
        if sample_count.item() == 0:
            raise RuntimeError("MMAudio validation did not produce any samples")
        loss = float((loss_sum / sample_count).item())
        if self._rank == 0:
            tracker = getattr(method, "tracker", None)
            if tracker is not None:
                tracker.log({"validation/flow_matching_loss": loss}, iteration)
            logger.info("MMAudio validation step=%d loss=%.8f", iteration, loss)
        return first_batch

    def _build_pipeline(self, transformer: torch.nn.Module) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        from fastvideo.pipelines.basic.mmaudio import MMAudioPipeline

        ignored_component = object()
        self._pipeline = MMAudioPipeline.from_pretrained(
            self.inference_model_path,
            inference_mode=True,
            loaded_modules={
                "transformer": transformer,
                "text_encoder": ignored_component,
                "tokenizer": ignored_component,
                "image_encoder": ignored_component,
                "image_encoder_2": ignored_component,
            },
            workload_type="v2a",
            num_gpus=int(self.training_config.distributed.num_gpus),
            tp_size=int(self.training_config.distributed.tp_size),
            sp_size=int(self.training_config.distributed.sp_size),
            hsdp_shard_dim=int(self.training_config.distributed.hsdp_shard_dim),
            pin_cpu_memory=bool(self.training_config.distributed.pin_cpu_memory),
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            vae_cpu_offload=False,
        )
        self._pipeline.fastvideo_args.pipeline_config = self.training_config.pipeline_config
        return self._pipeline

    @torch.inference_mode()
    def _run_inference(
        self,
        method: TrainingMethod,
        raw_batch: dict[str, Any],
        iteration: int,
    ) -> None:
        pipeline = self._build_pipeline(method.student.transformer)
        pc = self.training_config.pipeline_config
        if pc is None:
            raise RuntimeError("MMAudio inference validation requires pipeline_config")

        text_features = raw_batch.get("text_features")
        if not isinstance(text_features, torch.Tensor):
            raise RuntimeError("MMAudio validation inference requires text features")
        available = int(text_features.shape[0])
        if available <= 0:
            raise RuntimeError("MMAudio validation inference batch is empty")
        world_size = (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
        assignments = _global_inference_indices(
            self.inference_num_samples,
            world_size,
            self._rank,
        )
        output_root = Path(self.output_dir or
                           self.training_config.checkpoint.output_dir, ) / "validation_audio" / f"step_{iteration:09d}"
        output_root.mkdir(parents=True, exist_ok=True)

        for local_index, global_index in enumerate(assignments):
            # Every FSDP rank must execute the same number of forwards. Ranks
            # outside a non-divisible global budget run one padded call but do
            # not save it.
            sample_index = local_index % available
            output_index = (global_index if global_index is not None else self.inference_num_samples + self._rank)
            batch = ForwardBatch(
                data_type="video",
                prompt="",
                negative_prompt="",
                audio_start_in_s=0.0,
                audio_end_in_s=float(pc.duration_s),
                num_inference_steps=self.inference_num_steps,
                guidance_scale=self.inference_guidance_scale,
                seed=self.inference_seed + output_index,
                num_videos_per_prompt=1,
                height=8,
                width=8,
                num_frames=1,
                save_video=False,
                return_frames=False,
                extra={
                    "mmaudio_clip_features": raw_batch["clip_features"][sample_index:sample_index + 1],
                    "mmaudio_sync_features": raw_batch["sync_features"][sample_index:sample_index + 1],
                    "mmaudio_text_features": raw_batch["text_features"][sample_index:sample_index + 1],
                },
            )
            result = pipeline.forward(batch, pipeline.fastvideo_args)
            audio = result.extra.get("audio")
            sample_rate = result.extra.get("audio_sample_rate")
            if not isinstance(audio, np.ndarray) or not isinstance(sample_rate, int):
                raise RuntimeError("MMAudio validation inference did not produce decoded audio")
            if global_index is None:
                continue
            sample_id = self._metadata_value(raw_batch, "sample_id", sample_index)
            source_path = self._metadata_value(raw_batch, "source_path", sample_index)
            stem = f"sample_{global_index:03d}_{_safe_sample_name(sample_id)}"
            audio_path = output_root / f"{stem}.wav"
            wavfile.write(audio_path, sample_rate, audio.astype(np.float32))
            logger.info("Saved MMAudio validation audio to %s", audio_path)
            if self.inference_save_video and source_path:
                self._mux_source_video(
                    source_path=Path(source_path),
                    output_path=output_root / f"{stem}.mp4",
                    audio=audio,
                    sample_rate=sample_rate,
                    duration_s=float(pc.duration_s),
                )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self._rank == 0 and self.inference_log_to_tracker:
            self._log_inference_media(method, output_root, iteration)

    @staticmethod
    def _metadata_value(
        raw_batch: dict[str, Any],
        key: str,
        index: int,
    ) -> str:
        values = raw_batch.get(key)
        if isinstance(values, list | tuple) and index < len(values):
            return str(values[index])
        return ""

    @staticmethod
    def _mux_source_video(
        *,
        source_path: Path,
        output_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        duration_s: float,
    ) -> Path | None:
        if not source_path.is_file():
            logger.warning("Cannot compose MMAudio validation video; missing %s", source_path)
            return None
        try:
            import av
            from av import AudioFrame

            # Match official mmaudio.data.av_utils.read_frames: keep every
            # decoded source frame through duration_s at the stream's guessed
            # frame rate, including a frame exactly on the endpoint.
            all_frames: list[np.ndarray] = []
            with av.open(str(source_path)) as source:
                input_stream = source.streams.video[0]
                frame_rate = input_stream.guessed_rate
                input_stream.thread_type = "AUTO"
                for packet in source.demux(input_stream):
                    reached_end = False
                    for frame in packet.decode():
                        if frame.time is not None and frame.time > duration_s:
                            reached_end = True
                            break
                        all_frames.append(frame.to_ndarray(format="rgb24"))
                    if reached_end:
                        break
            if not all_frames:
                raise ValueError("source video produced no decoded frames")

            # Match official reencode_with_audio: H.264 at the original
            # guessed FPS, 10 Mbps, yuv420p, followed by mono float audio
            # encoded as AAC at the model sample rate.
            output = av.open(str(output_path), "w")
            try:
                video_stream = output.add_stream("h264", frame_rate)
                video_stream.codec_context.bit_rate = int(10 * 1e6)
                video_stream.width = int(all_frames[0].shape[1])
                video_stream.height = int(all_frames[0].shape[0])
                video_stream.pix_fmt = "yuv420p"
                audio_stream = output.add_stream("aac", sample_rate)

                for frame_array in all_frames:
                    video_frame = av.VideoFrame.from_ndarray(frame_array)
                    for packet in video_stream.encode(video_frame):
                        output.mux(packet)
                for packet in video_stream.encode():
                    output.mux(packet)

                audio_np = np.asarray(audio, dtype=np.float32)
                if audio_np.ndim == 1:
                    audio_np = audio_np[None, :]
                elif audio_np.ndim == 2 and audio_np.shape[1] == 1:
                    audio_np = audio_np.T
                if audio_np.ndim != 2 or audio_np.shape[0] != 1:
                    raise ValueError(f"Expected mono audio, got {audio_np.shape}")
                audio_frame = AudioFrame.from_ndarray(
                    np.ascontiguousarray(audio_np),
                    format="flt",
                    layout="mono",
                )
                audio_frame.sample_rate = sample_rate
                for packet in audio_stream.encode(audio_frame):
                    output.mux(packet)
                for packet in audio_stream.encode():
                    output.mux(packet)
            finally:
                output.close()
        except Exception as exc:
            logger.warning(
                "Failed to compose MMAudio validation video %s: %s",
                output_path,
                exc,
            )
            return None
        logger.info("Saved MMAudio validation video to %s", output_path)
        return output_path

    def _log_inference_media(
        self,
        method: TrainingMethod,
        output_root: Path,
        iteration: int,
    ) -> None:
        tracker = getattr(method, "tracker", None)
        if tracker is None:
            return
        artifacts = []
        for path in sorted(output_root.glob("sample_*.mp4"))[:self.inference_num_samples]:
            artifact = tracker.video(
                str(path),
                caption=path.stem,
                format="mp4",
            )
            if artifact is not None:
                artifacts.append(artifact)
        if artifacts:
            tracker.log_artifacts(
                {"validation/mmaudio_samples": artifacts},
                iteration,
            )

    def on_train_end(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        del method, iteration
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None


__all__ = ["MMAudioValidationCallback"]
