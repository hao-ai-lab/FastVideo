# SPDX-License-Identifier: Apache-2.0
"""I2V (image-to-video) input construction for Wan2.2-TI2V-5B on MLX.

TI2V-5B has no CLIP image embedder. Image conditioning is purely:

1. VAE-encode the input image to a single latent frame (caller / torch-MPS).
2. **Replace** ``noise_latents[:, :, 0]`` with that image latent.
3. Build a **per-token** timestep ``[B, L]`` with frame-0 tokens at ``t=0``
   (clean) and the remaining tokens at the denoise level.

Token order is **frame-major**, matching ``MLXWan22DiT._patch_embed`` and the
torch Wan patch embed: for each post-patch frame, tokens scan ``H_p × W_p``.
So the first ``tokens_per_frame`` entries of the ``[B, L]`` timestep are the
image-frame tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


def tokens_per_frame(height: int, width: int, patch_size: Sequence[int]) -> int:
    """Number of patch tokens in one latent frame (frame-major layout)."""
    _pt, ph, pw = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    if height % ph != 0 or width % pw != 0:
        raise ValueError(f"latent HxW ({height}x{width}) not divisible by patch {(ph, pw)}")
    return (height // ph) * (width // pw)


def num_patch_tokens(frames: int, height: int, width: int, patch_size: Sequence[int]) -> int:
    """Total sequence length ``L`` after patch embedding."""
    pt, _, _ = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    if frames % pt != 0:
        raise ValueError(f"latent frames {frames} not divisible by patch_t {pt}")
    return (frames // pt) * tokens_per_frame(height, width, patch_size)


def replace_first_latent_frame(noise_latents: Any, image_latent_frame: Any) -> Any:
    """Return latents with frame 0 replaced by ``image_latent_frame``.

    Args:
        noise_latents: ``[B, C, T, H, W]`` (torch or numpy or mx.array).
        image_latent_frame: ``[B, C, H, W]`` or ``[B, C, 1, H, W]``.
    """
    # Dispatch without importing mlx/torch at module import time.
    if not isinstance(noise_latents, np.ndarray):
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is not None and isinstance(noise_latents, torch.Tensor):
            out = noise_latents.clone()
            img = image_latent_frame
            if img.dim() == 5:
                img = img[:, :, 0]
            if tuple(img.shape) != tuple(out[:, :, 0].shape):
                raise ValueError(f"image frame shape {tuple(img.shape)} != "
                                 f"latent frame 0 shape {tuple(out[:, :, 0].shape)}")
            out[:, :, 0] = img.to(dtype=out.dtype, device=out.device)
            return out

        import mlx.core as mx

        if isinstance(noise_latents, mx.array):
            img = image_latent_frame
            if not isinstance(img, mx.array):
                img = mx.array(np.asarray(img))
            if img.ndim == 4:
                img = img[:, :, None, :, :]
            if int(img.shape[2]) != 1:
                raise ValueError(f"image_latent_frame must be a single frame, got {img.shape}")
            frame0_shape = tuple(noise_latents[:, :, :1, :, :].shape)
            if tuple(img.shape) != frame0_shape:
                raise ValueError(f"image frame shape {tuple(img.shape)} != latent frame 0 shape {frame0_shape}")
            rest = noise_latents[:, :, 1:, :, :]
            return mx.concatenate([img.astype(noise_latents.dtype), rest], axis=2)

    # NumPy path.
    out = np.array(noise_latents, copy=True)
    img = np.asarray(image_latent_frame)
    if img.ndim == 5:
        img = img[:, :, 0]
    if img.shape != out[:, :, 0].shape:
        raise ValueError(f"image frame shape {img.shape} != latent frame 0 shape {out[:, :, 0].shape}")
    out[:, :, 0] = img
    return out


def build_i2v_per_token_timestep(
    *,
    batch: int,
    frames: int,
    height: int,
    width: int,
    patch_size: Sequence[int],
    video_timestep: float,
    image_timestep: float = 0.0,
    as_numpy: bool = True,
) -> Any:
    """Build frame-major per-token timesteps for TI2V-style I2V.

    Frame 0's tokens get ``image_timestep`` (default 0 = clean image lock);
    remaining frames get ``video_timestep``.

    Returns:
        Array of shape ``[batch, L]`` as NumPy float32 by default, or an
        ``mx.array`` when ``as_numpy=False``.
    """
    tpf = tokens_per_frame(height, width, patch_size)
    n_frames_patched = frames // int(patch_size[0])
    levels = [float(image_timestep)] + [float(video_timestep)] * (n_frames_patched - 1)
    flat = [levels[i // tpf] for i in range(n_frames_patched * tpf)]
    arr = np.array([flat] * batch, dtype=np.float32)
    if as_numpy:
        return arr
    import mlx.core as mx

    return mx.array(arr)


def build_i2v_inputs(
        noise_latents: Any,
        image_latent_frame: Any,
        *,
        video_timestep: float,
        image_timestep: float = 0.0,
        patch_size: Sequence[int] = (1, 2, 2),
) -> tuple[Any, np.ndarray]:
    """Construct ``(latents, per_token_timestep)`` for one I2V DiT forward.

    Works with NumPy / torch / mx latents; timestep is always returned as
    NumPy float32 ``[B, L]`` (caller casts for the backend).
    """
    shape = tuple(noise_latents.shape) if hasattr(noise_latents, "shape") else np.asarray(noise_latents).shape
    if len(shape) != 5:
        raise ValueError(f"noise_latents must be [B,C,T,H,W], got {shape}")
    batch, _c, frames, height, width = shape
    latents = replace_first_latent_frame(noise_latents, image_latent_frame)
    timestep = build_i2v_per_token_timestep(
        batch=batch,
        frames=frames,
        height=height,
        width=width,
        patch_size=patch_size,
        video_timestep=video_timestep,
        image_timestep=image_timestep,
        as_numpy=True,
    )
    return latents, timestep
