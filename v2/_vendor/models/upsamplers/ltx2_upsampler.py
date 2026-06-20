# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 latent upsampler (spatial/temporal) implementation.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class PixelShuffleND(nn.Module):
    """N-dimensional pixel shuffle for upsampling."""

    def __init__(self, dims: int, upscale_factors: Tuple[int, int, int] = (2, 2, 2)) -> None:
        super().__init__()
        if dims not in (1, 2, 3):
            raise ValueError("dims must be 1, 2, or 3")
        self.dims = dims
        self.upscale_factors = upscale_factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 3:
            return rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
                p3=self.upscale_factors[2],
            )
        if self.dims == 2:
            return rearrange(
                x,
                "b (c p1 p2) h w -> b c (h p1) (w p2)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
            )
        if self.dims == 1:
            return rearrange(
                x,
                "b (c p1) f h w -> b c (f p1) h w",
                p1=self.upscale_factors[0],
            )
        raise ValueError(f"Unsupported dims: {self.dims}")


class BlurDownsample(nn.Module):
    """
    Anti-aliased spatial downsampling by integer stride using a fixed separable binomial kernel.
    Applies only on H,W. Works for dims=2 or dims=3 (per-frame).
    """

    def __init__(self, dims: int, stride: int, kernel_size: int = 5) -> None:
        super().__init__()
        if dims not in (2, 3):
            raise ValueError("dims must be 2 or 3")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be an odd integer >= 3")

        self.dims = dims
        self.stride = stride
        self.kernel_size = kernel_size

        k = torch.tensor([math.comb(kernel_size - 1, idx) for idx in range(kernel_size)])
        k2d = k[:, None] @ k[None, :]
        k2d = (k2d / k2d.sum()).float()
        self.register_buffer("kernel", k2d[None, None, :, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x
        if self.dims == 2:
            return self._apply_2d(x)
        b, _, f, _, _ = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self._apply_2d(x)
        h2, w2 = x.shape[-2:]
        return rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f, h=h2, w=w2)

    def _apply_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        c = x2d.shape[1]
        weight = self.kernel.expand(c, 1, self.kernel_size, self.kernel_size)
        return nn.functional.conv2d(
            x2d,
            weight=weight,
            bias=None,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=c,
        )


def _rational_for_scale(scale: float) -> Tuple[int, int]:
    mapping = {0.75: (3, 4), 1.5: (3, 2), 2.0: (2, 1), 4.0: (4, 1)}
    if float(scale) not in mapping:
        raise ValueError(f"Unsupported scale {scale}. Choose from {list(mapping.keys())}")
    return mapping[float(scale)]


class SpatialRationalResampler(nn.Module):
    """
    Fully-learned rational spatial scaling: up by 'num' via PixelShuffle, then
    anti-aliased downsample by 'den' using fixed blur + stride. Operates on H,W only.
    For dims==3, work per-frame for spatial scaling (temporal axis untouched).
    """

    def __init__(self, mid_channels: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.num, self.den = _rational_for_scale(self.scale)
        self.conv = nn.Conv2d(mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = BlurDownsample(dims=2, stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        return rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)


class ResBlock(nn.Module):
    """Residual block with two convolutional layers, group norm, and SiLU."""

    def __init__(self, channels: int, mid_channels: Optional[int] = None, dims: int = 3) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.conv1 = conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, mid_channels)
        self.conv2 = conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return x


class LatentUpsampler(nn.Module):
    """
    Model to upsample VAE latents spatially and/or temporally.
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.initial_conv = conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)
        self.initial_activation = nn.SiLU()

        self.res_blocks = nn.ModuleList([ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)])

        if spatial_upsample and temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            if rational_resampler:
                self.upsampler = SpatialRationalResampler(mid_channels=mid_channels, scale=self.spatial_scale)
            else:
                self.upsampler = nn.Sequential(
                    nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    PixelShuffleND(2),
                )
        elif temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        self.post_upsample_res_blocks = nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = conv(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                x = x[:, :, 1:, :, :]
            elif isinstance(self.upsampler, SpatialRationalResampler):
                x = self.upsampler(x)
            else:
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x


class LatentUpsamplerConfigurator:
    """Configurator for LatentUpsampler from a config dict."""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LatentUpsampler:
        cfg = dict(config)
        cfg.pop("_class_name", None)
        if "upsampler" in cfg and isinstance(cfg["upsampler"], dict):
            cfg = cfg["upsampler"]

        return LatentUpsampler(
            in_channels=cfg.get("in_channels", 128),
            mid_channels=cfg.get("mid_channels", 512),
            num_blocks_per_stage=cfg.get("num_blocks_per_stage", 4),
            dims=cfg.get("dims", 3),
            spatial_upsample=cfg.get("spatial_upsample", True),
            temporal_upsample=cfg.get("temporal_upsample", False),
            spatial_scale=cfg.get("spatial_scale", 2.0),
            rational_resampler=cfg.get("rational_resampler", False),
        )


class LTX2LatentUpsampler(nn.Module):
    """Public wrapper for the LTX-2 latent upsampler."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.model: LatentUpsampler = LatentUpsamplerConfigurator.from_config(config)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model(latent)


def upsample_video(latent: torch.Tensor, video_encoder: Any, upsampler: LatentUpsampler) -> torch.Tensor:
    """
    Upsample a latent tensor with normalization based on the video encoder's per-channel statistics.
    """
    if not hasattr(video_encoder, "per_channel_statistics"):
        raise ValueError("video_encoder must expose per_channel_statistics for normalization")
    stats = video_encoder.per_channel_statistics
    latent = stats.un_normalize(latent)
    latent = upsampler(latent)
    latent = stats.normalize(latent)
    return latent


__all__ = [
    "PixelShuffleND",
    "BlurDownsample",
    "SpatialRationalResampler",
    "ResBlock",
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "LTX2LatentUpsampler",
    "upsample_video",
]
