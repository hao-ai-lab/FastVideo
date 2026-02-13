#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos 2.5 / Wan2.1 VAE adapter.

Why this exists:
- Cosmos2.5 uses a Wan2.1-style VAE, but the *diffusion model* operates in a
  **normalized latent space**:
    z_norm = (z - mean) / std

  Meanwhile, FastVideo's `AutoencoderKLWan` operates in the VAE's native latent
  space (denormalized):
    z = z_norm * std + mean

This adapter provides a single, stable interface for FastVideo pipelines:
- `encode(x)` returns an object with `.mean` / `.sample()` / `.mode()`
- `decode(z)` returns a tensor in pixel space

It also exposes flags used by pipeline stages to avoid double (de)normalization:
- `handles_latent_norm = True`   -> stages should NOT normalize encoder latents
- `handles_latent_denorm = True` -> stages should NOT denormalize before decode
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class _TensorLatentDist:
    """Minimal distribution-like wrapper used by pipeline stages."""

    mean: torch.Tensor

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: Any | None = None) -> torch.Tensor:  # generator for API compatibility
        # The official interface encodes deterministically; for compatibility we
        # return the mean. (Stochastic posterior sampling isn't required for
        # Cosmos2.5 inference.)
        _ = generator
        return self.mean


class Cosmos25WanVAEAdapter(nn.Module):
    """
    Adapter that makes a Wan2.1-style VAE follow Cosmos2.5's latent contract:
    - `encode()` returns **normalized** latents
    - `decode()` expects **normalized** latents
    """

    # Pipeline stage hints (see latent_preparation.py / decoding.py / image_encoding.py)
    handles_latent_norm: bool = True
    handles_latent_denorm: bool = True
    latent_norm_mode: str = "internal"  # informational

    def __init__(
        self,
        inner: Any,
        *,
        latents_mean: torch.Tensor | None = None,
        latents_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.inner = inner

        # Preserve `config` when available; some pipeline utilities expect it.
        self.config = getattr(inner, "config", None)

        # If not provided, try to derive from `config.latents_mean/std`.
        cfg = self.config
        if latents_mean is None and cfg is not None and hasattr(cfg, "latents_mean"):
            latents_mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
        if latents_std is None and cfg is not None and hasattr(cfg, "latents_std"):
            latents_std = torch.tensor(cfg.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)

        if latents_mean is None or latents_std is None:
            raise RuntimeError(
                "Cosmos25WanVAEAdapter requires latents_mean/latents_std (either passed explicitly or available on inner.config)."
            )

        # Register as buffers so `.to(...)` moves them with the module.
        self.register_buffer("_latents_mean", latents_mean, persistent=False)
        self.register_buffer("_latents_std", latents_std, persistent=False)

    def _to_latent_stats(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._latents_mean.to(device=like.device, dtype=like.dtype)
        std = self._latents_std.to(device=like.device, dtype=like.dtype)
        return mean, std

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        # Keep parity with official interface.
        if hasattr(self.inner, "get_latent_num_frames"):
            return int(self.inner.get_latent_num_frames(num_pixel_frames))
        return 1 + (num_pixel_frames - 1) // 4

    def encode(self, x: torch.Tensor) -> _TensorLatentDist:
        """
        Returns *normalized* latents (Cosmos contract).
        """
        enc_out = self.inner.encode(x)

        # Support common encoder output shapes:
        # - DiagonalGaussianDistribution (FastVideo VAE): has `.mean` / `.sample()` / `.mode()`
        # - diffusers EncoderOutput: has `.latent_dist`
        # - raw tensor
        if hasattr(enc_out, "latent_dist"):
            dist = enc_out.latent_dist
            z_mean = dist.mode() if hasattr(dist, "mode") else dist.mean
        elif hasattr(enc_out, "mode") and hasattr(enc_out, "mean"):
            z_mean = enc_out.mode()
        elif isinstance(enc_out, torch.Tensor):
            z_mean = enc_out
        else:
            attrs = [a for a in dir(enc_out) if not a.startswith("_")]
            raise RuntimeError(
                f"Unsupported VAE encoder output type: {type(enc_out)}. attrs={attrs}"
            )

        mean, std = self._to_latent_stats(z_mean)
        z_norm = (z_mean - mean) / std
        return _TensorLatentDist(z_norm)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Expects *normalized* latents (Cosmos contract).
        """
        mean, std = self._to_latent_stats(z)
        z_denorm = z * std + mean
        out = self.inner.decode(z_denorm)
        return out.sample if hasattr(out, "sample") else out


#
# Official-like Wan2.1 VAE implementation (ported from cosmos_predict2 wan2pt1.py)
# -------------------------------------------------------------------------------
# Motivation:
# - We already solved checkpoint *key mapping* and can load official weights.
# - Remaining output drift vs the official tokenizer is largely decoder-side.
# - FastVideo's `AutoencoderKLWan` uses a different temporal upsample path
#   (`DupUp3D` + `first_chunk` slicing), while the official tokenizer uses
#   `Resample(mode="upsample3d")` with a time-conv + interleave reshape.
#
# This section ports the core modules (CausalConv3d/Resample/etc.) so we can run
# a VAE that is behaviorally closer to the official implementation WITHOUT
# importing any official repo classes at runtime.
#

CACHE_T = 2


class Cosmos25CausalConv3d(nn.Conv3d):
    """
    Official-like causal 3D convolution.

    Matches `CausalConv3d` in the official tokenizer: uses explicit F.pad and
    supports a `cache_x` prefix for causal chunking.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # padding order for F.pad: (W_left, W_right, H_left, H_right, T_left, T_right)
        self._padding: tuple[int, ...] = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class Cosmos25RMSNorm(nn.Module):
    """Official-like RMS_norm (uses learnable gamma and optional bias)."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma + self.bias


class Cosmos25Upsample(nn.Upsample):
    """Official-like Upsample that is safe for bf16 (casts to fp32 internally)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(x.float()).type_as(x)


class Cosmos25Resample(nn.Module):
    """
    Official-like Resample used for both spatial and temporal up/downsampling.
    """

    def __init__(self, dim: int, mode: str) -> None:
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Cosmos25Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Cosmos25Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = Cosmos25CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = Cosmos25CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: list[Any] | None = None, feat_idx: list[int] = [0]) -> torch.Tensor:
        b, c, t, h, w = x.size()

        # Temporal upsample uses a time-conv and then interleaves frames.
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)

                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        # Temporal downsample: time_conv consumes last-frame cache.
        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

        return x


class Cosmos25ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            Cosmos25RMSNorm(in_dim, images=False),
            nn.SiLU(),
            Cosmos25CausalConv3d(in_dim, out_dim, 3, padding=1),
            Cosmos25RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            Cosmos25CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = Cosmos25CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: list[Any] | None = None, feat_idx: list[int] = [0]) -> torch.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, Cosmos25CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class Cosmos25AttentionBlock(nn.Module):
    """Official-like causal self-attention with a single head."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = Cosmos25RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x2 = rearrange(x, "b c t h w -> (b t) c h w")
        x2 = self.norm(x2)
        q, k, v = (
            self.to_qkv(x2)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x2 = F.scaled_dot_product_attention(q, k, v)
        x2 = x2.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x2 = self.proj(x2)
        x2 = rearrange(x2, "(b t) c h w-> b c t h w", t=t)
        return x2 + identity


class Cosmos25Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 32,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temperal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = Cosmos25CausalConv3d(3, dims[0], 3, padding=1)

        downsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(Cosmos25ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(Cosmos25AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Cosmos25Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            Cosmos25ResidualBlock(out_dim, out_dim, dropout),
            Cosmos25AttentionBlock(out_dim),
            Cosmos25ResidualBlock(out_dim, out_dim, dropout),
        )

        self.head = nn.Sequential(
            Cosmos25RMSNorm(out_dim, images=False),
            nn.SiLU(),
            Cosmos25CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, feat_cache: list[Any] | None = None, feat_idx: list[int] = [0]) -> torch.Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)  # type: ignore[misc]
            else:
                x = layer(x)  # type: ignore[misc]

        for layer in self.middle:
            if isinstance(layer, Cosmos25ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)  # type: ignore[misc]

        for layer in self.head:
            if isinstance(layer, Cosmos25CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)  # type: ignore[misc]
        return x


class Cosmos25Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temperal_upsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = Cosmos25CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            Cosmos25ResidualBlock(dims[0], dims[0], dropout),
            Cosmos25AttentionBlock(dims[0]),
            Cosmos25ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(Cosmos25ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(Cosmos25AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Cosmos25Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            Cosmos25RMSNorm(out_dim, images=False),
            nn.SiLU(),
            Cosmos25CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, feat_cache: list[Any] | None = None, feat_idx: list[int] = [0]) -> torch.Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, Cosmos25ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)  # type: ignore[misc]

        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)  # type: ignore[misc]
            else:
                x = layer(x)  # type: ignore[misc]

        for layer in self.head:
            if isinstance(layer, Cosmos25CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)  # type: ignore[misc]
        return x


def _count_cosmos25_conv3d(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, Cosmos25CausalConv3d))


class Cosmos25WanVAE(nn.Module):
    """
    A FastVideo-native copy of the *official-like* Wan2.1 VAE core.

    Key properties:
    - Module naming matches official tokenizer (`encoder`, `decoder`, `conv1`, `conv2`)
      so it can consume `tokenizer.pth` keys directly.
    - `encode()` returns **normalized** latents and `decode()` expects **normalized**
      latents (Cosmos2.5 contract), matching `Wan2pt1VAEInterface`.
    """

    handles_latent_norm: bool = True
    handles_latent_denorm: bool = True

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        temporal_window: int = 4,
        latents_mean: torch.Tensor | None = None,
        latents_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        # Official hyperparams for Cosmos2.5 tokenizer (Wan2.1 VAE).
        cfg = dict(
            dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
            temporal_window=temporal_window,
        )
        self.z_dim = 16
        self.temporal_window = temporal_window

        self.encoder = Cosmos25Encoder3d(
            dim=cfg["dim"],
            z_dim=cfg["z_dim"] * 2,
            dim_mult=cfg["dim_mult"],
            num_res_blocks=cfg["num_res_blocks"],
            attn_scales=cfg["attn_scales"],
            temperal_downsample=cfg["temperal_downsample"],
            dropout=cfg["dropout"],
        )
        self.conv1 = Cosmos25CausalConv3d(self.z_dim * 2, self.z_dim * 2, 1)
        self.conv2 = Cosmos25CausalConv3d(self.z_dim, self.z_dim, 1)
        self.decoder = Cosmos25Decoder3d(
            dim=cfg["dim"],
            z_dim=cfg["z_dim"],
            dim_mult=cfg["dim_mult"],
            num_res_blocks=cfg["num_res_blocks"],
            attn_scales=cfg["attn_scales"],
            temperal_upsample=list(cfg["temperal_downsample"])[::-1],
            dropout=cfg["dropout"],
        )

        # Default Cosmos2.5 latent stats (shared with configs).
        if latents_mean is None:
            latents_mean = torch.tensor(
                [
                    -0.7571,
                    -0.7089,
                    -0.9113,
                    0.1075,
                    -0.1745,
                    0.9653,
                    -0.1517,
                    1.5508,
                    0.4134,
                    -0.0715,
                    0.5517,
                    -0.3632,
                    -0.1922,
                    -0.9497,
                    0.2503,
                    -0.2921,
                ],
                dtype=torch.float32,
            ).view(1, 16, 1, 1, 1)
        if latents_std is None:
            latents_std = torch.tensor(
                [
                    2.8184,
                    1.4541,
                    2.3275,
                    2.6558,
                    1.2196,
                    1.7708,
                    2.6052,
                    2.0743,
                    3.2687,
                    2.1526,
                    2.8652,
                    1.5579,
                    1.6382,
                    1.1253,
                    2.8251,
                    1.9160,
                ],
                dtype=torch.float32,
            ).view(1, 16, 1, 1, 1)

        self.register_buffer("_latents_mean", latents_mean, persistent=False)
        self.register_buffer("_latents_std", latents_std, persistent=False)

        self.to(device=device, dtype=dtype)
        self.clear_cache()

    def clear_cache(self) -> None:
        # Decoder cache
        self._conv_num = _count_cosmos25_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map: list[Any] = [None] * self._conv_num
        # Encoder cache
        self._enc_conv_num = _count_cosmos25_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map: list[Any] = [None] * self._enc_conv_num

    def _scale(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._latents_mean.to(device=like.device, dtype=like.dtype)
        std = self._latents_std.to(device=like.device, dtype=like.dtype)
        return mean, 1.0 / std

    def _i0_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)

    def _i0_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x[:, :, 0:1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)

    def encode(self, x: torch.Tensor) -> _TensorLatentDist:
        """
        Encode to *normalized* latents (Cosmos contract).
        """
        self.clear_cache()
        t = x.shape[2]
        iters = 1 + (t - 1) // self.temporal_window

        for i in range(iters):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self._i0_encode(x)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + self.temporal_window * (i - 1) : 1 + self.temporal_window * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        if (t - 1) % self.temporal_window:
            self._enc_conv_idx = [0]
            out_ = self.encoder(
                x[:, :, 1 + self.temporal_window * (iters - 1) :, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)

        mu, _log_var = self.conv1(out).chunk(2, dim=1)
        mean, inv_std = self._scale(mu)
        z_norm = (mu - mean) * inv_std
        self.clear_cache()
        return _TensorLatentDist(z_norm)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode from *normalized* latents (Cosmos contract).
        """
        self.clear_cache()
        mean, inv_std = self._scale(latent)
        z = latent / inv_std + mean  # z = z_norm * std + mean

        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self._i0_decode(x)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)

        self.clear_cache()
        return out

    # --- Interface helpers (match official Wan2pt1VAEInterface) ---
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (int(num_pixel_frames) - 1) // 4

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (int(num_latent_frames) - 1) * 4 + 1

    @property
    def spatial_compression_factor(self) -> int:
        return 8

    @property
    def temporal_compression_factor(self) -> int:
        return 4

    @property
    def latent_ch(self) -> int:
        return 16









