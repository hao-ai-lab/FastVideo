# SPDX-License-Identifier: Apache-2.0
"""Inference stages for MMAudio V2A/T2A.

The video sampler intentionally mirrors ``mmaudio.data.av_utils.read_frames``:
timestamps are sampled independently at 8 FPS and 25 FPS, and a decoded frame
is repeated when the source FPS is lower than a requested sampling rate.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _read_frames_at_fps(
    video_path: str | Path,
    frame_rates: tuple[float, ...],
    *,
    start_s: float,
    end_s: float,
) -> list[np.ndarray]:
    """Decode RGB frames with MMAudio's timestamp/duplication contract."""
    import av

    outputs: list[list[np.ndarray]] = [[] for _ in frame_rates]
    next_times = [0.0 for _ in frame_rates]
    deltas = [1.0 / fps for fps in frame_rates]

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_time = frame.time
                if frame_time is None or frame_time < start_s:
                    continue
                if frame_time > end_s:
                    break

                frame_array = None
                for index in range(len(frame_rates)):
                    while frame_time >= next_times[index]:
                        if frame_array is None:
                            frame_array = frame.to_ndarray(format="rgb24")
                        outputs[index].append(frame_array)
                        next_times[index] += deltas[index]

    if any(not frames for frames in outputs):
        raise ValueError(
            f"Could not decode enough video frames from {video_path} in "
            f"[{start_s}, {end_s}] seconds."
        )
    return [np.stack(frames) for frames in outputs]


def preprocess_mmaudio_video(
    video_path: str | Path,
    *,
    duration_s: float,
    clip_fps: int = 8,
    sync_fps: int = 25,
    clip_size: int = 384,
    sync_size: int = 224,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return official-format CLIP frames, sync frames, and effective duration.

    CLIP output is float32 ``[T,3,384,384]`` in ``[0,1]``. Synchformer
    output is float32 ``[T,3,224,224]`` in ``[-1,1]``.
    """
    clip_array, sync_array = _read_frames_at_fps(
        video_path,
        (float(clip_fps), float(sync_fps)),
        start_s=0.0,
        end_s=duration_s,
    )
    clip_frames = torch.from_numpy(clip_array).permute(0, 3, 1, 2)
    sync_frames = torch.from_numpy(sync_array).permute(0, 3, 1, 2)

    clip_transform = v2.Compose(
        [
            v2.Resize((clip_size, clip_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    sync_transform = v2.Compose(
        [
            v2.Resize(sync_size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(sync_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    clip_frames = clip_transform(clip_frames)
    sync_frames = sync_transform(sync_frames)

    effective_duration = min(
        duration_s,
        clip_frames.shape[0] / clip_fps,
        sync_frames.shape[0] / sync_fps,
    )
    clip_frames = clip_frames[: int(clip_fps * effective_duration)]
    sync_frames = sync_frames[: int(sync_fps * effective_duration)]
    return clip_frames, sync_frames, effective_duration


def mmaudio_sequence_lengths(duration_s: float, pc) -> tuple[int, int, int]:
    latent_length = math.ceil(
        duration_s * pc.sampling_rate / pc.spectrogram_frame_rate / pc.latent_downsample_rate
    )
    clip_length = int(duration_s * pc.clip_frame_rate)
    sync_frame_count = duration_s * pc.sync_frame_rate
    sync_segments = (
        (sync_frame_count - pc.sync_segment_size) // pc.sync_segment_stride + 1
    )
    sync_length = int(sync_segments * pc.sync_segment_size / pc.sync_downsample_rate)
    return latent_length, clip_length, sync_length


class MMAudioInputValidationStage(PipelineStage):
    """Validate audio-generation inputs without invoking video pipeline logic."""

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.num_inference_steps <= 0:
            raise ValueError("MMAudio num_inference_steps must be positive.")
        if batch.num_videos_per_prompt != 1:
            raise ValueError("MMAudio currently supports one output per request.")
        if isinstance(batch.prompt, list) and len(batch.prompt) != 1:
            raise ValueError("MMAudio currently supports one prompt per request.")
        if isinstance(batch.negative_prompt, list) and len(batch.negative_prompt) != 1:
            raise ValueError("MMAudio currently supports one negative prompt per request.")

        direct_video = (
            batch.extra.get("mmaudio_clip_frames") is not None
            and batch.extra.get("mmaudio_sync_frames") is not None
        )
        if (
            fastvideo_args.workload_type is WorkloadType.V2A
            and batch.video_path is None
            and not direct_video
        ):
            raise ValueError("MMAudio V2A requires `video_path` or preprocessed MMAudio frame tensors.")

        pc = fastvideo_args.pipeline_config
        duration_s = batch.audio_end_in_s
        duration_s = pc.duration_s if duration_s is None else float(duration_s)
        start_s = 0.0 if batch.audio_start_in_s is None else float(batch.audio_start_in_s)
        if start_s != 0.0:
            raise ValueError("MMAudio currently generates from time zero; audio_start_in_s must be 0.")
        max_duration_s = pc.max_audio_duration_s
        if max_duration_s is not None and duration_s > max_duration_s:
            raise ValueError(
                f"MMAudio duration {duration_s}s exceeds this checkpoint's "
                f"{max_duration_s}s maximum."
            )
        minimum_duration = pc.sync_segment_size / pc.sync_frame_rate
        if duration_s < minimum_duration:
            raise ValueError(
                f"MMAudio needs at least {minimum_duration:.2f}s "
                f"({pc.sync_segment_size} sync frames), got {duration_s}s."
            )
        batch.extra["mmaudio_duration_s"] = duration_s
        batch.seed = 0 if batch.seed is None else int(batch.seed)
        return batch


class MMAudioVideoConditioningStage(PipelineStage):
    """Decode video and run DFN5B/Synchformer with official preprocessing."""

    def __init__(self, image_encoder, sync_encoder, transformer) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.sync_encoder = sync_encoder
        self.transformer = transformer

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        pc = fastvideo_args.pipeline_config
        duration_s = float(batch.extra["mmaudio_duration_s"])
        clip_frames = batch.extra.get("mmaudio_clip_frames")
        sync_frames = batch.extra.get("mmaudio_sync_frames")
        use_video = batch.video_path is not None or (clip_frames is not None and sync_frames is not None)

        if batch.video_path is not None:
            clip_frames, sync_frames, duration_s = preprocess_mmaudio_video(
                batch.video_path,
                duration_s=duration_s,
                clip_fps=pc.clip_frame_rate,
                sync_fps=pc.sync_frame_rate,
                clip_size=pc.clip_image_size,
                sync_size=pc.sync_image_size,
            )
        elif use_video:
            if clip_frames.ndim == 5:
                if clip_frames.shape[0] != 1:
                    raise ValueError("MMAudio preprocessed CLIP frames must have batch size 1.")
                clip_frames = clip_frames[0]
            if sync_frames.ndim == 5:
                if sync_frames.shape[0] != 1:
                    raise ValueError("MMAudio preprocessed sync frames must have batch size 1.")
                sync_frames = sync_frames[0]
            duration_s = min(
                duration_s,
                clip_frames.shape[0] / pc.clip_frame_rate,
                sync_frames.shape[0] / pc.sync_frame_rate,
            )
            clip_frames = clip_frames[: int(duration_s * pc.clip_frame_rate)]
            sync_frames = sync_frames[: int(duration_s * pc.sync_frame_rate)]

        latent_length, clip_length, sync_length = mmaudio_sequence_lengths(duration_s, pc)
        if clip_length <= 0 or sync_length <= 0:
            raise ValueError(f"MMAudio duration {duration_s}s produces an empty condition sequence.")
        self.transformer.update_seq_lengths(latent_length, clip_length, sync_length)
        batch.extra["mmaudio_duration_s"] = duration_s
        batch.extra["mmaudio_sequence_lengths"] = (latent_length, clip_length, sync_length)

        if not use_video:
            batch.extra["mmaudio_clip_features"] = self.transformer.get_empty_clip_sequence(1)
            batch.extra["mmaudio_sync_features"] = self.transformer.get_empty_sync_sequence(1)
            return batch

        assert clip_frames is not None and sync_frames is not None
        if clip_frames.shape != (clip_length, 3, pc.clip_image_size, pc.clip_image_size):
            raise ValueError(
                "MMAudio CLIP frames must have shape "
                f"[{clip_length},3,{pc.clip_image_size},{pc.clip_image_size}], got {tuple(clip_frames.shape)}."
            )
        expected_sync_frames = int(duration_s * pc.sync_frame_rate)
        if sync_frames.shape != (expected_sync_frames, 3, pc.sync_image_size, pc.sync_image_size):
            raise ValueError(
                "MMAudio sync frames must have shape "
                f"[{expected_sync_frames},3,{pc.sync_image_size},{pc.sync_image_size}], "
                f"got {tuple(sync_frames.shape)}."
            )

        device = get_local_torch_device()
        model_dtype = next(self.transformer.parameters()).dtype

        self.image_encoder = self.image_encoder.to(device)
        clip_video = clip_frames.to(device=device, dtype=model_dtype, non_blocking=True)
        mean = torch.tensor(_CLIP_MEAN, device=device, dtype=model_dtype).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=device, dtype=model_dtype).view(1, 3, 1, 1)
        clip_video = (clip_video - mean) / std
        clip_outputs: list[torch.Tensor] = []
        chunk_size = pc.clip_batch_size_multiplier
        with set_forward_context(current_timestep=0, attn_metadata=None):
            for start in range(0, clip_length, chunk_size):
                encoded = self.image_encoder(clip_video[start : start + chunk_size]).last_hidden_state
                clip_outputs.append(encoded)
        clip_features = torch.cat(clip_outputs, dim=0).unsqueeze(0)
        if fastvideo_args.image_encoder_cpu_offload:
            self.image_encoder = self.image_encoder.to("cpu")

        self.sync_encoder = self.sync_encoder.to(device)
        sync_video = sync_frames.to(device=device, dtype=model_dtype, non_blocking=True).unsqueeze(0)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            sync_features = self.sync_encoder(sync_video).last_hidden_state
        if fastvideo_args.image_encoder_cpu_offload:
            self.sync_encoder = self.sync_encoder.to("cpu")

        if sync_features.shape[1] != sync_length:
            raise RuntimeError(
                f"Synchformer produced {sync_features.shape[1]} tokens; expected {sync_length}."
            )
        batch.extra["mmaudio_clip_features"] = clip_features
        batch.extra["mmaudio_sync_features"] = sync_features
        return batch


class MMAudioTextConditioningStage(PipelineStage):
    """Encode positive/negative OpenCLIP token sequences and project conditions."""

    def __init__(self, text_encoder, tokenizer, transformer) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = get_local_torch_device()
        model_dtype = next(self.transformer.parameters()).dtype
        self.text_encoder = self.text_encoder.to(device)

        def encode(text: str | list[str] | None) -> torch.Tensor | None:
            if text is None:
                return None
            texts = [text] if isinstance(text, str) else text
            tokens = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            # Hugging Face CLIPTokenizer pads with EOS by default, while
            # OpenCLIP/MMAudio pads with token id zero.
            input_ids = tokens.input_ids.masked_fill(tokens.attention_mask == 0, 0).to(device)
            with set_forward_context(current_timestep=0, attn_metadata=None):
                return self.text_encoder(input_ids).last_hidden_state.to(model_dtype)

        text_features = encode(batch.prompt)
        if text_features is None:
            text_features = self.transformer.get_empty_string_sequence(1)
        negative_text_features = encode(batch.negative_prompt)

        if fastvideo_args.text_encoder_cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")

        conditions = self.transformer.preprocess_conditions(
            batch.extra["mmaudio_clip_features"],
            batch.extra["mmaudio_sync_features"],
            text_features,
        )
        empty_conditions = self.transformer.get_empty_conditions(
            1,
            negative_text_features=negative_text_features,
        )
        batch.extra["mmaudio_conditions"] = conditions
        batch.extra["mmaudio_empty_conditions"] = empty_conditions
        return batch


class MMAudioLatentPreparationStage(PipelineStage):
    """Sample the Gaussian flow prior using a device-local seeded generator."""

    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.latents is not None:
            return batch
        device = get_local_torch_device()
        dtype = next(self.transformer.parameters()).dtype
        generator = torch.Generator(device=device).manual_seed(int(batch.seed))
        batch.generator = generator
        batch.latents = torch.randn(
            (1, self.transformer.latent_seq_len, self.transformer.latent_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return batch


class MMAudioDenoisingStage(PipelineStage):
    """Run MMAudio's forward-time Euler flow with FastVideo's shared scheduler."""

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.latents is not None
        # Official MMAudio builds ``torch.linspace`` on CPU. A CPU scalar in
        # the bf16 CUDA update follows scalar-promotion rules, whereas moving
        # the same float32 scalar to CUDA changes the rounded trajectory.
        self.scheduler.set_timesteps(batch.num_inference_steps, device="cpu")
        conditions = batch.extra["mmaudio_conditions"]
        empty_conditions = batch.extra["mmaudio_empty_conditions"]
        latents = batch.latents
        for index, timestep in enumerate(self.scheduler.timesteps):
            flow = self.transformer.guided_flow(
                timestep / self.scheduler.config.num_train_timesteps,
                latents,
                conditions,
                empty_conditions,
                float(batch.guidance_scale),
            )
            # The shared FlowMatch scheduler supplies the exact inverted
            # forward-time sigma schedule. Its generic ``step`` deliberately
            # upcasts samples to fp32, however, while MMAudio's published
            # Euler loop performs ``x += dt * flow`` in the model's bf16
            # dtype. Preserve that operation order/precision here; changing
            # it shifts the full 25-step trajectory.
            delta = self.scheduler.sigmas[index + 1] - self.scheduler.sigmas[index]
            latents = latents + delta * flow
            batch.step_index = index
            batch.timestep = timestep
        batch.latents = self.transformer.unnormalize(latents)
        return batch


class MMAudioDecodingStage(PipelineStage):
    """Decode MMAudio latents to a mono 44.1 kHz waveform."""

    def __init__(self, audio_vae, vocoder) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.vocoder = vocoder

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.latents is not None
        if fastvideo_args.output_type == "latent":
            batch.output = batch.latents.detach().cpu()
            return batch

        device = get_local_torch_device()
        self.audio_vae = self.audio_vae.to(device)
        self.vocoder = self.vocoder.to(device)
        decoder_dtype = next(self.audio_vae.parameters()).dtype
        mel = self.audio_vae.decode(batch.latents.transpose(1, 2).to(decoder_dtype))
        audio = self.vocoder(mel.to(next(self.vocoder.parameters()).dtype))

        pc = fastvideo_args.pipeline_config
        expected_samples = batch.extra["mmaudio_sequence_lengths"][0]
        expected_samples *= pc.spectrogram_frame_rate * pc.latent_downsample_rate
        if audio.shape[-1] != expected_samples:
            raise RuntimeError(
                f"MMAudio vocoder produced {audio.shape[-1]} samples; expected {expected_samples}."
            )

        decoded_audio = audio.detach().float().cpu()
        batch.extra["audio"] = decoded_audio[0].T.contiguous().numpy()
        batch.extra["audio_sample_rate"] = int(pc.sampling_rate)
        batch.extra["audio_only"] = True
        batch.extra["decoded_audio"] = decoded_audio
        batch.data_type = "audio"

        # VideoGenerator currently transports a tensor for every workload;
        # audio-only paths never materialize or save these pixels.
        batch.output = torch.zeros((1, 3, 1, 8, 8), dtype=torch.uint8)
        return batch
