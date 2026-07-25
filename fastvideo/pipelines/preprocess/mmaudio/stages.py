# SPDX-License-Identifier: Apache-2.0
"""MMAudio-specific feature stages for the shared FastVideo V2A workflow."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import torch
import torchaudio

from fastvideo.configs.configs import VideoLoaderType
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.mmaudio.stages import (
    _CLIP_MEAN,
    _CLIP_STD,
    preprocess_mmaudio_video,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, PreprocessBatch
from fastvideo.pipelines.preprocess.mmaudio.torio_media_reader import (PREPROCESSED_MEDIA_KEY, PREPROCESS_ERROR_KEY,
                                                                       preprocess_mmaudio_media_with_torio)
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class MMAudioFeatureExtractionStage(PipelineStage):
    """Extract VAE posterior statistics plus CLIP/Synchformer features."""

    def __init__(
        self,
        *,
        audio_vae: torch.nn.Module,
        mel_converter: torch.nn.Module,
        image_encoder: torch.nn.Module,
        sync_encoder: torch.nn.Module,
        text_encoder: torch.nn.Module,
        tokenizer: Any,
    ) -> None:
        super().__init__()
        self.audio_vae = audio_vae.eval()
        self.mel_converter = mel_converter.eval()
        self.image_encoder = image_encoder.eval()
        self.sync_encoder = sync_encoder.eval()
        self.text_encoder = text_encoder.eval()
        self.tokenizer = tokenizer

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @staticmethod
    def _load_audio(
        path: str | Path,
        *,
        target_sample_rate: int,
        target_samples: int,
        normalize_audio: bool,
    ) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.ndim != 2 or waveform.shape[0] == 0:
            raise ValueError(f"Decoded audio has an invalid shape: {tuple(waveform.shape)}")
        waveform = waveform.mean(dim=0)
        if normalize_audio:
            abs_max = waveform.abs().max()
            if abs_max <= 1e-6:
                raise ValueError("Decoded audio is silent")
            waveform = waveform / abs_max * 0.95
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                target_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
        if waveform.shape[0] < target_samples:
            raise ValueError(
                f"Audio is too short: need {target_samples} samples at {target_sample_rate} Hz, got {waveform.shape[0]}"
            )
        return waveform[:target_samples].contiguous()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        batch = cast(PreprocessBatch, batch)
        if not isinstance(batch.prompt, list):
            raise ValueError("MMAudio preprocessing expects a list of captions")
        if len(batch.video_loader) != len(batch.prompt):
            raise ValueError("MMAudio preprocessing paths and captions have different batch sizes")

        pc = fastvideo_args.pipeline_config
        duration_s = float(pc.duration_s)
        sample_rate = int(pc.sampling_rate)
        alignment = int(pc.spectrogram_frame_rate * pc.latent_downsample_rate)
        target_samples = math.ceil(duration_s * sample_rate / alignment) * alignment
        expected_clip_frames = int(duration_s * pc.clip_frame_rate)
        expected_sync_frames = int(duration_s * pc.sync_frame_rate)
        preprocess_config = fastvideo_args.preprocess_config
        dataset_split = "train" if preprocess_config is None else preprocess_config.dataset_split.lower()
        normalize_audio = dataset_split == "train"
        use_torio = (preprocess_config is not None and preprocess_config.video_loader_type == VideoLoaderType.TORIO)

        preprocessed_media = batch.extra.get(PREPROCESSED_MEDIA_KEY)
        preprocess_errors = batch.extra.get(PREPROCESS_ERROR_KEY)
        for name, values in (
            (PREPROCESSED_MEDIA_KEY, preprocessed_media),
            (PREPROCESS_ERROR_KEY, preprocess_errors),
        ):
            if values is not None and (not isinstance(values, list) or len(values) != len(batch.video_loader)):
                raise ValueError(f"MMAudio {name} must align with the input batch")

        valid_ids: list[str] = []
        valid_paths: list[str] = []
        valid_captions: list[str] = []
        audio_batch: list[torch.Tensor] = []
        clip_batch: list[torch.Tensor] = []
        sync_batch: list[torch.Tensor] = []
        failures: list[dict[str, str]] = []

        for index, (sample_id, path,
                    caption) in enumerate(zip(
                        batch.video_file_name,
                        batch.video_loader,
                        batch.prompt,
                        strict=True,
                    )):
            try:
                path_str = str(path)
                if not Path(path_str).is_file():
                    raise FileNotFoundError(path_str)
                preprocess_error = (None if preprocess_errors is None else preprocess_errors[index])
                if preprocess_error is not None:
                    raise ValueError(str(preprocess_error))
                media = (None if preprocessed_media is None else preprocessed_media[index])
                if media is not None:
                    if not isinstance(media, dict):
                        raise ValueError("Preprocessed MMAudio media must be a dictionary")
                    audio = media["audio"]
                    clip_frames = media["clip_frames"]
                    sync_frames = media["sync_frames"]
                    effective_duration = float(media["effective_duration"])
                elif use_torio:
                    audio, clip_frames, sync_frames, effective_duration = (preprocess_mmaudio_media_with_torio(
                        path_str,
                        duration_s=duration_s,
                        target_sample_rate=sample_rate,
                        target_samples=target_samples,
                        normalize_audio=normalize_audio,
                        clip_fps=pc.clip_frame_rate,
                        sync_fps=pc.sync_frame_rate,
                        clip_size=pc.clip_image_size,
                        sync_size=pc.sync_image_size,
                    ))
                else:
                    audio = self._load_audio(
                        path_str,
                        target_sample_rate=sample_rate,
                        target_samples=target_samples,
                        normalize_audio=normalize_audio,
                    )
                    clip_frames, sync_frames, effective_duration = preprocess_mmaudio_video(
                        path_str,
                        duration_s=duration_s,
                        clip_fps=pc.clip_frame_rate,
                        sync_fps=pc.sync_frame_rate,
                        clip_size=pc.clip_image_size,
                        sync_size=pc.sync_image_size,
                        use_ffmpeg_fps_filter=True,
                    )
                if effective_duration < duration_s:
                    raise ValueError(f"Video is too short: need {duration_s}s, got {effective_duration:.3f}s")
                if clip_frames.shape[0] != expected_clip_frames:
                    raise ValueError(f"Expected {expected_clip_frames} CLIP frames, got {clip_frames.shape[0]}")
                if sync_frames.shape[0] != expected_sync_frames:
                    raise ValueError(f"Expected {expected_sync_frames} sync frames, got {sync_frames.shape[0]}")
            except Exception as exc:
                logger.warning("Skipping MMAudio preprocessing sample %s: %s", sample_id, exc)
                failures.append({"id": str(sample_id), "error": str(exc)})
                continue
            valid_ids.append(str(sample_id))
            valid_paths.append(path_str)
            valid_captions.append(str(caption))
            audio_batch.append(audio)
            clip_batch.append(clip_frames)
            sync_batch.append(sync_frames)

        batch.extra["failed_samples"] = failures
        if not valid_ids:
            batch.video_file_name = []
            batch.video_loader = []
            batch.prompt = []
            batch.extra["precomputed_features"] = {}
            batch.extra["precomputed_metadata"] = []
            return batch

        device = get_local_torch_device()
        self.audio_vae = self.audio_vae.to(device=device, dtype=torch.float32)
        self.mel_converter = self.mel_converter.to(device=device, dtype=torch.float32)
        waveforms = torch.stack(audio_batch).to(device=device, dtype=torch.float32)
        posterior = self.audio_vae.encode(self.mel_converter(waveforms))
        latent_mean = posterior.mean.transpose(1, 2).detach().cpu()
        latent_std = posterior.std.transpose(1, 2).detach().cpu()

        self.image_encoder = self.image_encoder.to(device=device, dtype=torch.float32)
        clip_video = torch.stack(clip_batch).to(device=device, dtype=torch.float32)
        mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=device).view(1, 1, 3, 1, 1)
        clip_video = ((clip_video - mean) / std).flatten(0, 1)
        clip_outputs: list[torch.Tensor] = []
        chunk_size = int(pc.clip_batch_size_multiplier)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            for start in range(0, clip_video.shape[0], chunk_size):
                encoded = self.image_encoder(clip_video[start:start + chunk_size]).last_hidden_state
                clip_outputs.append(encoded)
        clip_features = torch.cat(clip_outputs).reshape(len(valid_ids), expected_clip_frames, -1).detach().cpu()

        self.sync_encoder = self.sync_encoder.to(device=device, dtype=torch.float32)
        sync_video = torch.stack(sync_batch).to(device=device, dtype=torch.float32)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            sync_features = self.sync_encoder(sync_video).last_hidden_state.detach().cpu()

        self.text_encoder = self.text_encoder.to(device=device, dtype=torch.float32)
        tokens = self.tokenizer(
            valid_captions,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.masked_fill(tokens.attention_mask == 0, 0).to(device)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            text_features = self.text_encoder(input_ids).last_hidden_state.detach().cpu()

        batch.video_file_name = valid_ids
        batch.video_loader = valid_paths
        batch.prompt = valid_captions
        batch.extra["precomputed_features"] = {
            "mean": latent_mean,
            "std": latent_std,
            "clip_features": clip_features,
            "sync_features": sync_features,
            "text_features": text_features,
        }
        batch.extra["precomputed_metadata"] = [{
            "id": sample_id,
            "caption": caption,
            "source": path,
            "split": dataset_split,
        } for sample_id, caption, path in zip(
            valid_ids,
            valid_captions,
            valid_paths,
            strict=True,
        )]
        return batch


__all__ = ["MMAudioFeatureExtractionStage"]
