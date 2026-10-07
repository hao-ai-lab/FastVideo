# SPDX-License-Identifier: Apache-2.0
"""Pixel-space spatial resampling for decoded MLX Wan frames.

Spatial fast mode denoises on a smaller latent grid and has to get back to
the requested output size. That resize belongs *here* — after the VAE
decode — and not in latent space.

A Wan latent cell is a learned code for an 8x8 (Wan2.1) or 16x16 (Wan2.2)
pixel block, not a low-pass sample of the image. Linearly blending two
adjacent codes does not produce the code of the blended blocks; it produces
a vector the decoder was never trained on. The decoder answers with smeared,
ringing texture laid over otherwise-correct structure — the silhouette
survives, the detail turns to haze. Measured on Wan2.1-1.3B at 480x832, a
2x bilinear latent upsample destroys 62% of the latent's high-frequency
energy while leaving its overall magnitude intact, which is exactly the
signature of that veil.

Resampling decoded RGB frames has none of that problem: an image *is* a
sampled 2-D signal, so Lanczos/cubic interpolation is the operation it was
defined for. The result is soft — it carries stage-1's real detail budget
and no more — but it is clean and coherent.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# Pixel-space interpolation kernels, best-quality first. ``lanczos`` is the
# default: it holds edges better than cubic at 2x with no visible ringing on
# decoder output, which is already band-limited.
PIXEL_UPSAMPLE_MODES = ("lanczos", "cubic", "bilinear", "nearest")

DEFAULT_PIXEL_UPSAMPLE_MODE = "lanczos"


def _interpolation_flag(mode: str) -> int:
    """
    Map a pixel upsample mode name onto its OpenCV interpolation flag.

    Parameters:
        mode (str): One of :data:`PIXEL_UPSAMPLE_MODES`.

    Returns:
        int: The matching ``cv2.INTER_*`` flag.

    Raises:
        ValueError: If the mode is not a supported pixel upsample mode.
    """
    import cv2

    flags = {
        "lanczos": cv2.INTER_LANCZOS4,
        "cubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }
    try:
        return flags[mode]
    except KeyError:
        raise ValueError(f"Unsupported pixel upsample mode: {mode!r} "
                         f"(expected one of {', '.join(PIXEL_UPSAMPLE_MODES)})") from None


def unsharp(frame: np.ndarray, amount: float) -> np.ndarray:
    """Light unsharp mask, used to counter resampling / optical-flow softening.

    Parameters:
        frame (np.ndarray): HxWx3 uint8 RGB frame.
        amount (float): Strength; ``0`` returns the frame unchanged.

    Returns:
        np.ndarray: A new frame; the input is never modified in place.
    """
    if amount <= 0.0:
        return frame
    import cv2

    blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def upsample_frame(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    mode: str = DEFAULT_PIXEL_UPSAMPLE_MODE,
    sharpen: float = 0.0,
) -> np.ndarray:
    """
    Resample one decoded RGB frame to the target pixel size.

    Parameters:
        frame (np.ndarray): HxWx3 uint8 RGB frame.
        width (int): Target width in pixels.
        height (int): Target height in pixels.
        mode (str): Interpolation kernel, one of :data:`PIXEL_UPSAMPLE_MODES`.
        sharpen (float): Unsharp strength applied after the resize.

    Returns:
        np.ndarray: A new frame at ``height x width``; already-correct sizes
            are still passed through ``sharpen``.

    Raises:
        ValueError: If the frame is not HxWx3, or the target size is not positive.
    """
    import cv2

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"frame must have shape HxWx3, got {array.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"target size must be positive, got {width}x{height}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if (array.shape[0], array.shape[1]) != (height, width):
        array = cv2.resize(array, (width, height), interpolation=_interpolation_flag(mode))
    return unsharp(array, sharpen)


def upsample_frames(
    frames: Iterable[np.ndarray],
    *,
    width: int,
    height: int,
    mode: str = DEFAULT_PIXEL_UPSAMPLE_MODE,
    sharpen: float = 0.0,
) -> list[np.ndarray]:
    """
    Resample every decoded frame to the target pixel size.

    Parameters:
        frames (Iterable[np.ndarray]): Decoded HxWx3 uint8 RGB frames.
        width (int): Target width in pixels.
        height (int): Target height in pixels.
        mode (str): Interpolation kernel, one of :data:`PIXEL_UPSAMPLE_MODES`.
        sharpen (float): Unsharp strength applied after each resize.

    Returns:
        list[np.ndarray]: New frames at the target size, in input order.
    """
    return [upsample_frame(frame, width=width, height=height, mode=mode, sharpen=sharpen) for frame in frames]


__all__ = [
    "DEFAULT_PIXEL_UPSAMPLE_MODE",
    "PIXEL_UPSAMPLE_MODES",
    "unsharp",
    "upsample_frame",
    "upsample_frames",
]
