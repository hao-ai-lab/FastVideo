# SPDX-License-Identifier: Apache-2.0
"""Optional torio media reader for MMAudio training preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torchvision.transforms import v2

PREPROCESSED_MEDIA_KEY = "preprocessed_media"
PREPROCESS_ERROR_KEY = "preprocess_error"


def preprocess_mmaudio_media_with_torio(
    media_path: str | Path,
    *,
    duration_s: float,
    target_sample_rate: int,
    target_samples: int,
    normalize_audio: bool,
    clip_fps: int = 8,
    sync_fps: int = 25,
    clip_size: int = 384,
    sync_size: int = 224,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Decode audio and dual-rate video with StreamingMediaDecoder.

    The dependency is imported lazily so normal FastVideo environments do not
    require the deprecated torio package. This backend is intended for offline
    reference-compatible training preprocessing only.
    """
    try:
        from torio.io import StreamingMediaDecoder
    except ImportError as exc:
        raise ImportError("MMAudio torio preprocessing requires the isolated torio "
                          "environment. Use run_preprocess_vggsound_torio.sh.") from exc

    expected_clip_frames = int(clip_fps * duration_s)
    expected_sync_frames = int(sync_fps * duration_s)
    reader = StreamingMediaDecoder(str(media_path))
    reader.add_basic_video_stream(
        frames_per_chunk=expected_clip_frames,
        frame_rate=float(clip_fps),
        format="rgb24",
    )
    reader.add_basic_video_stream(
        frames_per_chunk=expected_sync_frames,
        frame_rate=float(sync_fps),
        format="rgb24",
    )
    reader.add_basic_audio_stream(frames_per_chunk=2**30)
    reader.fill_buffer()
    clip_frames, sync_frames, audio_frames = reader.pop_chunks()

    if clip_frames is None:
        raise ValueError("CLIP video stream returned no frames")
    if clip_frames.shape[0] < expected_clip_frames:
        raise ValueError(f"CLIP video is too short: expected {expected_clip_frames} frames, "
                         f"got {clip_frames.shape[0]}")
    if sync_frames is None:
        raise ValueError("Synchformer video stream returned no frames")
    if sync_frames.shape[0] < expected_sync_frames:
        raise ValueError(f"Synchformer video is too short: expected {expected_sync_frames} "
                         f"frames, got {sync_frames.shape[0]}")
    if audio_frames is None or audio_frames.ndim != 2:
        shape = None if audio_frames is None else tuple(audio_frames.shape)
        raise ValueError(f"Decoded audio has an invalid shape: {shape}")

    waveform = audio_frames.transpose(0, 1).mean(dim=0)
    if normalize_audio:
        abs_max = waveform.abs().max()
        if abs_max <= 1e-6:
            raise ValueError("Decoded audio is silent")
        waveform = waveform / abs_max * 0.95

    source_sample_rate = int(reader.get_out_stream_info(2).sample_rate)
    if source_sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(
            source_sample_rate,
            target_sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )(waveform)
    if waveform.shape[0] < target_samples:
        raise ValueError(f"Audio is too short: need {target_samples} samples at "
                         f"{target_sample_rate} Hz, got {waveform.shape[0]}")
    waveform = waveform[:target_samples].contiguous()

    clip_transform = v2.Compose([
        v2.Resize(
            (clip_size, clip_size),
            interpolation=v2.InterpolationMode.BICUBIC,
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    sync_transform = v2.Compose([
        v2.Resize(
            sync_size,
            interpolation=v2.InterpolationMode.BICUBIC,
        ),
        v2.CenterCrop(sync_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    clip_frames = clip_transform(clip_frames[:expected_clip_frames])
    sync_frames = sync_transform(sync_frames[:expected_sync_frames])
    effective_duration = min(
        duration_s,
        clip_frames.shape[0] / clip_fps,
        sync_frames.shape[0] / sync_fps,
    )
    return waveform, clip_frames, sync_frames, effective_duration


@dataclass(frozen=True)
class MMAudioTorioRowPreprocessor:
    """Decode one V2A metadata row on a background CPU thread."""

    duration_s: float
    target_sample_rate: int
    target_samples: int
    normalize_audio: bool
    clip_fps: int = 8
    sync_fps: int = 25
    clip_size: int = 384
    sync_size: int = 224

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        output = dict(row)
        media_path = Path(str(output["video_path"]))
        try:
            if not media_path.is_file():
                raise FileNotFoundError(media_path)
            audio, clip, sync, duration = preprocess_mmaudio_media_with_torio(
                media_path,
                duration_s=self.duration_s,
                target_sample_rate=self.target_sample_rate,
                target_samples=self.target_samples,
                normalize_audio=self.normalize_audio,
                clip_fps=self.clip_fps,
                sync_fps=self.sync_fps,
                clip_size=self.clip_size,
                sync_size=self.sync_size,
            )
            output[PREPROCESSED_MEDIA_KEY] = {
                "audio": audio,
                "clip_frames": clip,
                "sync_frames": sync,
                "effective_duration": duration,
            }
        except Exception as exc:
            output[PREPROCESS_ERROR_KEY] = f"{type(exc).__name__}: {exc}"
        return output


__all__ = [
    "MMAudioTorioRowPreprocessor",
    "PREPROCESSED_MEDIA_KEY",
    "PREPROCESS_ERROR_KEY",
    "preprocess_mmaudio_media_with_torio",
]
