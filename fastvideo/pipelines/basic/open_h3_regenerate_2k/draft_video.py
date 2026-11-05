# SPDX-License-Identifier: Apache-2.0
"""Decode a draft video into the 124-frame pixel block the H3 VAE turns into 37 latent frames."""

from __future__ import annotations

import subprocess

import numpy as np
import torch

import fastvideo.envs as envs

DRAFT_SIZES: dict[str, tuple[int, int]] = {"768p": (1344, 768), "576p": (1024, 576)}
DRAFT_FRAMES = 121
# The causal VAE maps 124 frames to 37 latent frames; the last frame is held for the three extra ones.
VAE_PADDING_FRAMES = 3
# Seconds of held last frame ffmpeg appends so a short draft still yields DRAFT_FRAMES.
_TAIL_PAD_SECONDS = 6
_DECODE_TIMEOUT_S = 600


def ffmpeg_decode_command(path: str, width: int, height: int) -> list[str]:
    """fps=24, cover-scale (bilinear) and centre-crop, last frame held, exactly DRAFT_FRAMES frames, no audio."""
    video_filter = (f"fps=24,scale={width}:{height}:force_original_aspect_ratio=increase:flags=bilinear,"
                    f"crop={width}:{height},tpad=stop_mode=clone:stop_duration={_TAIL_PAD_SECONDS}")
    return [
        envs.FASTVIDEO_FFMPEG_BIN.get(), "-v", "error", "-i", path, "-vf", video_filter, "-frames:v",
        str(DRAFT_FRAMES), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
    ]


def load_draft_pixels(path: str, size: str = "768p") -> torch.Tensor:
    """uint8 pixels [1, 3, 124, H, W] on the CPU for the draft video at ``path``."""
    if size not in DRAFT_SIZES:
        raise ValueError(f"draft size must be one of {sorted(DRAFT_SIZES)}, got {size!r}.")
    width, height = DRAFT_SIZES[size]
    try:
        raw = subprocess.run(ffmpeg_decode_command(path, width, height),
                             capture_output=True,
                             check=True,
                             timeout=_DECODE_TIMEOUT_S).stdout
    except FileNotFoundError as error:
        raise RuntimeError(f"ffmpeg ({envs.FASTVIDEO_FFMPEG_BIN.get()!r}) is required to read the draft video; "
                           "set FASTVIDEO_FFMPEG_BIN.") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Could not decode the draft video {path!r}: {error.stderr.decode(errors='replace')}") from error
    expected = DRAFT_FRAMES * height * width * 3
    if len(raw) != expected:
        raise ValueError(f"The draft video {path!r} decoded to {len(raw)} bytes, expected {expected} "
                         f"({DRAFT_FRAMES} frames at {width}x{height}).")
    frames = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(DRAFT_FRAMES, height, width, 3).copy())
    pixels = frames.permute(3, 0, 1, 2)[None]
    return torch.cat([pixels, pixels[:, :, -1:].expand(-1, -1, VAE_PADDING_FRAMES, -1, -1)], dim=2).contiguous()
