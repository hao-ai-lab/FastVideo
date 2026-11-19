# SPDX-License-Identifier: Apache-2.0
"""Video byte encoding for reward uploads."""

from __future__ import annotations

import io

import numpy as np
import torch


def encode_video_bytes(media: torch.Tensor, *, fps: int = 16) -> bytes:
    """Encode one ``[C, T, H, W]`` float video in [0, 1] to mp4 bytes."""
    import imageio.v3 as iio

    if media.ndim == 5:
        if int(media.shape[0]) != 1:
            raise ValueError(f"expected a single video, got batch {media.shape[0]}")
        media = media[0]
    if media.ndim != 4:
        raise ValueError(f"media must be [C, T, H, W], got {tuple(media.shape)}")
    video = (media.detach().float().clamp(0, 1) * 255).round().to(torch.uint8)
    frames: np.ndarray = video.permute(1, 2, 3, 0).cpu().numpy()
    buffer = io.BytesIO()
    iio.imwrite(buffer, frames, extension=".mp4", fps=int(fps), codec="libx264")
    return buffer.getvalue()
