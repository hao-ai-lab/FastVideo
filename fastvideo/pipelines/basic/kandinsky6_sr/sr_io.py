# SPDX-License-Identifier: Apache-2.0
"""Video / audio input and delivery resizing for Kandinsky6 video super-resolution.

The SR model has its own input contract, so it does not go through the generic ``load_video`` (which decodes every
frame and drops the audio) or ``InputValidationStage`` (which resizes to the request geometry): the source is resampled
to 24 fps by a fixed stride, capped at 121 frames aligned to ``1 + 8k`` and kept at its own resolution.  Frames are
streamed and only the selected ones are kept; the audio track is decoded separately so it can be muxed back.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from fastvideo.logger import init_logger

logger = init_logger(__name__)

TARGET_FPS = 24
MAX_NUM_FRAMES = 121
# 23.98 and 25 fps sources are treated as already at the training rate.
RESAMPLE_FPS_TOLERANCE = 1.5
AUDIO_SAMPLE_RATE = 44100

#: Delivery tiers ``name -> (H, W)`` candidates (landscape, square, portrait); the closest aspect ratio wins.
TARGET_RESOLUTIONS: dict[str, tuple[tuple[int, int], ...]] = {
    "hd": ((720, 1280), (720, 720), (1280, 720)),
    "fullhd": ((1080, 1920), (1080, 1080), (1920, 1080)),
    "2k": ((1440, 2560), (1440, 1440), (2560, 1440)),
}


@dataclass(frozen=True)
class FramePlan:
    """Stream indices to keep and the rate they play at.

    When downsampling, the kept count is ``int(total_frames / step)``, which depends on the true stream length:
    ``lookahead`` frames must be read before the selection is final.
    """

    indices: tuple[int, ...]
    out_fps: int
    step: float | None = None
    lookahead: int = 0


def plan_frame_selection(src_fps: float) -> FramePlan:
    if src_fps <= 0:
        raise ValueError(f"Source video reports no usable fps (got {src_fps}).")
    if abs(src_fps - TARGET_FPS) < RESAMPLE_FPS_TOLERANCE:
        return FramePlan(tuple(range(MAX_NUM_FRAMES)), round(src_fps))
    if src_fps < TARGET_FPS:
        logger.warning("Source fps %.2f < %d: keeping the native frame rate (mildly out of distribution).", src_fps,
                       TARGET_FPS)
        return FramePlan(tuple(range(MAX_NUM_FRAMES)), round(src_fps))
    step = src_fps / TARGET_FPS
    virtual_total = math.ceil(MAX_NUM_FRAMES * step) + 1
    indices = [round(i * step) for i in range(int(virtual_total / step))]
    indices = [i for i in indices if i < virtual_total][:MAX_NUM_FRAMES]
    logger.warning("Source fps %.2f > %d: downsampling by a fixed stride.", src_fps, TARGET_FPS)
    return FramePlan(tuple(indices), TARGET_FPS, step, max(math.ceil(MAX_NUM_FRAMES * step), indices[-1] + 1))


def _select_frames(frames: Iterator[torch.Tensor], plan: FramePlan) -> list[torch.Tensor]:
    wanted = frozenset(plan.indices)
    stop_after = max(plan.indices[-1] + 1, plan.lookahead)
    kept: list[torch.Tensor] = []
    seen = 0
    for index, frame in enumerate(frames):
        if index in wanted:
            kept.append(frame)
        seen = index + 1
        if seen >= stop_after:
            break
    if plan.step is not None and seen < plan.lookahead:
        kept = kept[:int(seen / plan.step)]
    return kept


def read_video(path: str | Path) -> tuple[torch.Tensor, int]:
    """Decode ``path`` to ``([T, C, H, W] uint8, output_fps)`` following the SR input contract."""
    import av

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {path}")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"Video {path.name} has no video stream.")
        stream = container.streams.video[0]
        plan = plan_frame_selection(float(stream.average_rate or stream.base_rate or 0.0))
        decoded = (torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                   for frame in container.decode(stream))
        frames = _select_frames(decoded, plan)
    if not frames:
        raise ValueError(f"Video {path.name} has no readable frames.")
    aligned = 1 + 8 * ((min(len(frames), MAX_NUM_FRAMES) - 1) // 8)
    return torch.stack(frames[:aligned]).contiguous(), plan.out_fps


def read_audio(path: str | Path, num_frames: int, fps: int) -> torch.Tensor | None:
    """Mono float audio of ``path`` at :data:`AUDIO_SAMPLE_RATE`, trimmed to the span of ``num_frames`` frames.

    Returns ``None`` when there is no audio track or it cannot be decoded: audio is best effort and never fails the
    video.
    """
    import av

    keep = int(round(num_frames / fps * AUDIO_SAMPLE_RATE))
    chunks: list[np.ndarray] = []
    produced = 0
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return None
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=AUDIO_SAMPLE_RATE)
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
                    produced += chunks[-1].shape[0]
                if produced >= keep + AUDIO_SAMPLE_RATE:
                    break
            else:
                chunks.extend(resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(None))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not decode the audio of %s (%s); the output will have no audio.", path, exc)
        return None
    if not chunks:
        return None
    audio = np.clip(np.concatenate(chunks), -1.0, 1.0).astype(np.float32, copy=False)[:keep]
    return torch.from_numpy(audio.copy()) if audio.size else None


def validate_target_spec(spec: str | None, mode: str) -> None:
    """Reject a bad delivery spec before any model runs."""
    resolve_target_hw(spec, (1, 1), mode, validate_only=True)


def resolve_target_hw(spec: str | None,
                      sr_hw: tuple[int, int],
                      mode: str,
                      validate_only: bool = False) -> tuple[int, int] | None:
    """Delivery size for ``spec`` (a tier name or ``WxH``); ``None`` when the SR result is kept as is.

    ``fit`` shrinks isotropically into the bucket (never upscales, even sizes for the codec); ``exact`` returns the
    bucket itself.
    """
    if mode not in ("fit", "exact"):
        raise ValueError(f"sr_target_resize_mode must be 'fit' or 'exact', got {mode!r}")
    if spec is None or spec.lower() == "none":
        return None
    key = spec.lower()
    if key in TARGET_RESOLUTIONS:
        ratio = sr_hw[1] / sr_hw[0]
        bucket = min(TARGET_RESOLUTIONS[key], key=lambda hw: abs(hw[1] / hw[0] - ratio))
    else:
        parts = key.split("x")
        if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
            raise ValueError(f"sr_target_resolution must be one of {sorted(TARGET_RESOLUTIONS)} or WxH, got {spec!r}")
        bucket = (int(parts[1]), int(parts[0]))
    if mode == "exact" or validate_only:
        return bucket
    scale = min(bucket[0] / sr_hw[0], bucket[1] / sr_hw[1])
    if scale >= 1.0:
        logger.warning("SR result %dx%d already fits the %dx%d bucket; keeping it as is.", sr_hw[1], sr_hw[0],
                       bucket[1], bucket[0])
        return None
    return tuple(min(b, round(side * scale / 2) * 2) for b, side in zip(bucket, sr_hw, strict=True))  # type: ignore


def resize_video(video: torch.Tensor, target_hw: tuple[int, int], frame_chunk: int = 8) -> torch.Tensor:
    """Antialiased downscale of a ``[C, T, H, W]`` uint8 video to ``target_hw`` (no crop, never upscales)."""
    channels, frames, height, width = video.shape
    target_h, target_w = target_hw
    if (height, width) == (target_h, target_w):
        return video
    if target_h > height or target_w > width:
        raise ValueError(f"Target {target_w}x{target_h} exceeds the SR result {width}x{height}; pick a lower tier or "
                         "a higher sr_resolution_scale.")
    if abs((width / height) / (target_w / target_h) - 1) > 0.02:
        logger.warning("Resizing %dx%d to %dx%d changes the aspect ratio (no crop).", width, height, target_w, target_h)
    out = torch.empty((channels, frames, target_h, target_w), dtype=torch.uint8)
    for t0 in range(0, frames, frame_chunk):
        chunk = video[:, t0:t0 + frame_chunk].permute(1, 0, 2, 3).float()
        resized = functional.interpolate(chunk, size=target_hw, mode="bilinear", antialias=True, align_corners=False)
        out[:, t0:t0 + frame_chunk] = resized.clamp(0, 255).round().to(torch.uint8).permute(1, 0, 2, 3)
    return out
