# SPDX-License-Identifier: Apache-2.0
"""NVlabs fast 24-layer H3 VAE decoder (StreamRender-H3) as a drop-in decode adapter.

Opt-in: ``FASTVIDEO_H3_FAST_VAE_DECODER=<path to fast_h3_vae_noncausal_24l.safetensors>``.
The reference engine (``engine="ref"``) is the plain PyTorch decoder shipped in
``ref/Sana/models/Open-H3-Regenerate-2K/open_h3/fast_vae`` (SDPA over 16x16-token windows),
so it runs on any CUDA GPU, including GB10/sm_121, where the deployment engine's
FlashAttention-4 + quack path does not. It replaces only the video decoder; the
released 36-layer VAE stays the exact default.

The adapter exposes the subset of the released ``AutoencoderKLMiniMaxH3`` decode API the
decoding stage uses (``denormalize_latents``, ``decoded_pixel_shape``, ``decode_autocast_dtype``,
``decode_to_pixels``), so the stage body is unchanged apart from which object it drives.

Input contract (identical to the released decoder's, just transposed):
``denormalize_latents`` receives unpatchified normalized latents ``(B, C, T, H, W)`` and
returns ``(B, T, C, H, W)``; the fast decoder applies ``z * latents_std + latents_mean`` itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fastvideo.logger import init_logger

logger = init_logger(__name__)

# FastVAE's ImageNet pixel statistics (fast_vae/__init__.py), used to map the decoder's
# ImageNet-normalized output back to [0, 1] exactly as the released VAE's denormalize_pixels.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

# Native temporal geometry of the fast decoder: 5 latent frames per block, 4 output frames
# per latent frame, the first 3 frames of each block dropped (17 frames per block).
BLOCK = 5
PATCH_T = 4
FRAMES_TO_TRIM = 3
SPATIAL_COMPRESSION = 16


def _fast_vae_repo_root() -> Path:
    """Directory holding the ``fast_vae`` package (parent of the package dir)."""
    return Path(__file__).resolve().parents[4] / "ref/Sana/models/Open-H3-Regenerate-2K/open_h3"


class FastH3VaeDecoder(nn.Module):
    """Adapter exposing the released-VAE decode surface over NVlabs' fast decoder."""

    decode_autocast_dtype = torch.bfloat16

    def __init__(self, weights_path: str | Path, base_model_path: str | Path, engine: str = "ref") -> None:
        super().__init__()
        repo_root = _fast_vae_repo_root()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from fast_vae import FastVAE  # type: ignore[import-not-found]

        self.latent_channels = 24
        self.spatial_compression_ratio = SPATIAL_COMPRESSION
        self._vae = FastVAE(str(weights_path), str(base_model_path), engine=engine)
        # nn.Module registration keeps the decoder on the right device for the stage's
        # ``.to(device)`` calls and release bookkeeping.
        self.model = self._vae.model
        device = next(self.model.parameters()).device
        self.register_buffer("pixel_mean",
                             torch.tensor(PIXEL_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1).to(device),
                             persistent=False)
        self.register_buffer("pixel_std",
                             torch.tensor(PIXEL_STD, dtype=torch.float32).view(1, 3, 1, 1, 1).to(device),
                             persistent=False)

    # -- released-VAE-compatible surface ------------------------------------------------
    def maybe_compile(self) -> None:
        """Optionally torch.compile the decoder (default on for the fast decoder).

        The fast decoder runs whole-frame at 2K, so eager peaks at ~95 GiB on GB10;
        ``torch.compile(mode="default")`` fuses the casts/norms/partition glue and drops
        that to ~36 GiB while cutting the warm decode ~60 s -> ~37 s. The first call of a
        new shape pays the compile (~25 s). Set ``FASTVIDEO_H3_FAST_VAE_COMPILE=0`` to
        keep eager.
        """
        import os

        if os.environ.get("FASTVIDEO_H3_FAST_VAE_COMPILE", "1").strip() == "0":
            return
        mode = os.environ.get("FASTVIDEO_H3_FAST_VAE_COMPILE_MODE", "default").strip() or "default"
        compiled = torch.compile(self.model, mode=mode)
        self.model = compiled
        self._vae.model = compiled

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) normalized -> (B, T, C, H, W); denormalization is internal."""
        return latents.transpose(1, 2).contiguous()

    def decoded_pixel_shape(self, latent_shape: torch.Size | tuple[int, ...]) -> tuple[int, int, int, int, int]:
        if len(latent_shape) != 5:
            raise ValueError(f"Fast H3 VAE latents must be five-dimensional, got {tuple(latent_shape)}.")
        batch_size, latent_num_frames, channels, latent_height, latent_width = map(int, latent_shape)
        if channels != self.latent_channels:
            raise ValueError(f"Fast H3 VAE latents must have {self.latent_channels} channels, "
                             f"got {channels}.")
        num_blocks = (latent_num_frames + BLOCK - 1) // BLOCK
        decoded_num_frames = latent_num_frames * PATCH_T - FRAMES_TO_TRIM * num_blocks
        return (
            batch_size,
            3,
            decoded_num_frames,
            latent_height * SPATIAL_COMPRESSION,
            latent_width * SPATIAL_COMPRESSION,
        )

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) normalized latents -> (B, 3, F, 16H, 16W) pixels in [0, 1]."""
        pixels = self._vae.decode(z.float())
        mean = self.pixel_mean.to(pixels.device)
        std = self.pixel_std.to(pixels.device)
        return (pixels * std + mean).clamp_(0, 1)

    def decode_to_pixels(self, z: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Decode into a caller-owned CPU buffer, matching ``AutoencoderKLMiniMaxH3.decode_to_pixels``."""
        expected_shape = self.decoded_pixel_shape(z.shape)
        if (output.device.type != "cpu" or output.dtype not in (torch.float32, torch.uint8)
                or tuple(output.shape) != expected_shape):
            raise ValueError("`output` must be a CPU float32 or uint8 tensor with shape "
                             f"{expected_shape}, got device={output.device}, dtype={output.dtype}, "
                             f"shape={tuple(output.shape)}.")
        pixels = self.decode(z)
        if output.dtype == torch.uint8:
            pixels = pixels.mul_(255).clamp_(0, 255).to(torch.uint8)
        pixels = pixels.contiguous()
        for batch_index in range(pixels.shape[0]):
            for channel_index in range(pixels.shape[1]):
                output[batch_index, channel_index].copy_(pixels[batch_index, channel_index], non_blocking=False)
        torch.cuda.current_stream(z.device).synchronize()
        return output


def fast_vae_enabled() -> bool:
    import os

    return bool(os.environ.get("FASTVIDEO_H3_FAST_VAE_DECODER", "").strip())


__all__ = ["FastH3VaeDecoder", "fast_vae_enabled"]
