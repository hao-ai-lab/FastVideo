# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native single-frame RGBA autoencoder for Qwen-Image-2.1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from fastvideo.configs.models.vaes.base import VAEConfig
from fastvideo.configs.models.vaes.qwen_image21 import QwenImage21VAEConfig
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | list[torch.Generator] | None = None) -> torch.Tensor:
        if isinstance(generator, list):
            if len(generator) != self.mean.shape[0]:
                raise ValueError("A generator list must contain one generator per batch element.")
            noise = torch.cat([
                self._noise(self.mean[index:index + 1], item) for index, item in enumerate(generator)
            ])
        else:
            noise = self._noise(self.mean, generator)
        return self.mean + self.std * noise

    @staticmethod
    def _noise(sample: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        noise_device = sample.device if generator is None else generator.device
        return torch.randn(sample.shape, generator=generator, device=noise_device, dtype=sample.dtype).to(sample.device)

    def kl(self, other: DiagonalGaussianDistribution | None = None) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(1, device=self.mean.device, dtype=self.mean.dtype)
        if other is None:
            terms = self.mean.square() + self.var - 1.0 - self.logvar
        else:
            terms = ((self.mean - other.mean).square() + self.var) / other.var - 1.0 - self.logvar + other.logvar
        return 0.5 * terms.sum(dim=(1, 2, 3))

    def nll(self, sample: torch.Tensor, dims: tuple[int, ...] = (1, 2, 3)) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(1, device=self.mean.device, dtype=self.mean.dtype)
        return 0.5 * (math.log(2 * math.pi) + self.logvar + (sample - self.mean).square() / self.var).sum(dim=dims)


@dataclass
class AutoencoderKLOutput:
    latent_dist: DiagonalGaussianDistribution


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class QwenImage21AvgDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels:
            raise ValueError("Average shortcut channels must be divisible by the downsampling factor.")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        batch, channels, frames, height, width = x.shape
        x = x.view(batch, channels, frames // self.factor_t, self.factor_t,
                   height // self.factor_s, self.factor_s, width // self.factor_s, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(batch, channels * self.factor, frames // self.factor_t,
                   height // self.factor_s, width // self.factor_s)
        x = x.view(batch, self.out_channels, self.group_size, frames // self.factor_t,
                   height // self.factor_s, width // self.factor_s)
        return x.mean(dim=2)


class QwenImage21DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        factor = factor_t * factor_s * factor_s
        if out_channels * factor % in_channels:
            raise ValueError("Duplicate shortcut channels must be divisible by the upsampling factor.")
        self.repeats = out_channels * factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(x.shape[0], self.out_channels, self.factor_t, self.factor_s, self.factor_s,
                   x.shape[2], x.shape[3], x.shape[4])
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.shape[0], self.out_channels, x.shape[2] * self.factor_t,
                   x.shape[4] * self.factor_s, x.shape[6] * self.factor_s)
        return x[:, :, self.factor_t - 1:] if first_chunk else x


class QwenImage21CausalConv3d(nn.Conv2d):
    """Spatial convolution retaining the image VAE's singleton frame axis."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        self.padding = (0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError("Qwen-Image-2.1 spatial convolutions require exactly one frame.")
        return super().forward(F.pad(x.squeeze(2), self._padding)).unsqueeze(2)


class QwenImage21RMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcast_dims = (1, 1) if images else (1, 1, 1)
        shape = (dim, *broadcast_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upcast = x.dtype in (torch.float16, torch.bfloat16) or any(
            name in str(x.dtype) for name in ("float4_", "float8_")
        )
        normalized = F.normalize(x.float() if upcast else x, dim=1 if self.channel_first else -1).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class QwenImage21Upsample(nn.Upsample):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class QwenImage21Resample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None) -> None:
        super().__init__()
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                QwenImage21Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = QwenImage21CausalConv3d(dim, dim * 2, (1, 1), padding=(0, 0))
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            if mode == "downsample3d":
                self.time_conv = QwenImage21CausalConv3d(dim, dim, (1, 1), stride=(1, 1), padding=(0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        # Temporal convolutions have no preceding frame in this image model.
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = self.resample(x)
        return x.view(batch, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)


class QwenImage21ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = QwenImage21RMSNorm(in_dim, images=False)
        self.conv1 = QwenImage21CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImage21RMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImage21CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = QwenImage21CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv_shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + residual


class QwenImage21AttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = QwenImage21RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        qkv = self.to_qkv(self.norm(x)).reshape(batch * frames, 1, channels * 3, -1)
        q, k, v = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        # The VAE's one-head spatial attention has no distributed sequence layout.
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch * frames, channels, height, width)
        x = self.proj(x).view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + residual


class QwenImage21MidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([
            QwenImage21ResidualBlock(dim, dim, dropout),
            QwenImage21ResidualBlock(dim, dim, dropout),
        ])
        self.attentions = nn.ModuleList([QwenImage21AttentionBlock(dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class QwenImage21ResidualDownBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, num_res_blocks: int,
                 temporal_downsample: bool, down_flag: bool) -> None:
        super().__init__()
        self.avg_shortcut = QwenImage21AvgDown3D(in_dim, out_dim, 2 if temporal_downsample else 1,
                                               2 if down_flag else 1)
        self.resnets = nn.ModuleList([
            QwenImage21ResidualBlock(in_dim if index == 0 else out_dim, out_dim, dropout)
            for index in range(num_res_blocks)
        ])
        self.downsampler = QwenImage21Resample(
            out_dim, "downsample3d" if temporal_downsample else "downsample2d"
        ) if down_flag else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.clone()
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(residual)


class QwenImage21Encoder3d(nn.Module):
    def __init__(self, arch) -> None:
        super().__init__()
        dims = [arch.base_dim * factor for factor in [1, *arch.dim_mult]]
        self.conv_in = QwenImage21CausalConv3d(arch.in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        scale = 1.0
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            down_flag = index != len(arch.dim_mult) - 1
            temporal = arch.temperal_downsample[index] if down_flag else False
            if arch.is_residual:
                self.down_blocks.append(QwenImage21ResidualDownBlock(
                    in_dim, out_dim, arch.dropout, arch.num_res_blocks, temporal, down_flag
                ))
            else:
                for _ in range(arch.num_res_blocks):
                    self.down_blocks.append(QwenImage21ResidualBlock(in_dim, out_dim, arch.dropout))
                    if scale in arch.attn_scales:
                        self.down_blocks.append(QwenImage21AttentionBlock(out_dim))
                    in_dim = out_dim
                if down_flag:
                    mode = "downsample3d" if temporal else "downsample2d"
                    self.down_blocks.append(QwenImage21Resample(out_dim, mode))
                    scale /= 2.0
        self.mid_block = QwenImage21MidBlock(out_dim, arch.dropout)
        self.norm_out = QwenImage21RMSNorm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, arch.z_dim * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class QwenImage21ResidualUpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, arch, temporal_upsample: bool, up_flag: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([
            QwenImage21ResidualBlock(in_dim if index == 0 else out_dim, out_dim, arch.dropout)
            for index in range(arch.num_res_blocks + 1)
        ])
        self.avg_shortcut = QwenImage21DupUp3D(
            in_dim, out_dim, 2 if temporal_upsample else 1, 2
        ) if up_flag else None
        self.upsampler = QwenImage21Resample(
            out_dim, "upsample3d" if temporal_upsample else "upsample2d", upsample_out_dim=out_dim
        ) if up_flag else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.clone()
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(residual, first_chunk=True)
        return x


class QwenImage21UpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, arch, upsample_mode: str | None) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([
            QwenImage21ResidualBlock(in_dim if index == 0 else out_dim, out_dim, arch.dropout)
            for index in range(arch.num_res_blocks + 1)
        ])
        self.upsamplers = nn.ModuleList([QwenImage21Resample(out_dim, upsample_mode)]) if upsample_mode else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        return self.upsamplers[0](x) if self.upsamplers is not None else x


class QwenImage21Decoder3d(nn.Module):
    def __init__(self, arch) -> None:
        super().__init__()
        dim = arch.decoder_base_dim if arch.decoder_base_dim is not None else arch.base_dim
        dims = [dim * factor for factor in [arch.dim_mult[-1], *arch.dim_mult[::-1]]]
        temporal_upsample = arch.temperal_downsample[::-1]
        self.conv_in = QwenImage21CausalConv3d(arch.z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImage21MidBlock(dims[0], arch.dropout)
        self.up_blocks = nn.ModuleList()
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = index != len(arch.dim_mult) - 1
            temporal = temporal_upsample[index] if up_flag else False
            if arch.is_residual:
                self.up_blocks.append(QwenImage21ResidualUpBlock(in_dim, out_dim, arch, temporal, up_flag))
            else:
                if index > 0:
                    in_dim //= 2
                mode = ("upsample3d" if temporal else "upsample2d") if up_flag else None
                self.up_blocks.append(QwenImage21UpBlock(in_dim, out_dim, arch, mode))
        self.norm_out = QwenImage21RMSNorm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, arch.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    batch, channels, frames, height, width = x.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image height and width must be divisible by the VAE patch size.")
    x = x.view(batch, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(batch, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)


def _unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    batch, channels, frames, height, width = x.shape
    channels //= patch_size * patch_size
    x = x.view(batch, channels, patch_size, patch_size, frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return x.view(batch, channels, frames, height * patch_size, width * patch_size)


class AutoencoderKLQwenImage21(nn.Module):
    _supports_gradient_checkpointing = False
    _group_offload_block_modules = ["quant_conv", "post_quant_conv", "encoder", "decoder"]

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        if not config.load_encoder or not config.load_decoder:
            raise ValueError("Qwen-Image-2.1 requires the complete encoder and decoder checkpoint.")
        arch = config.arch_config
        self.z_dim = arch.z_dim
        self.latent_channels = arch.z_dim
        self.encoder = QwenImage21Encoder3d(arch)
        self.quant_conv = QwenImage21CausalConv3d(arch.z_dim * 2, arch.z_dim * 2, 1)
        self.post_quant_conv = QwenImage21CausalConv3d(arch.z_dim, arch.z_dim, 1)
        self.decoder = QwenImage21Decoder3d(arch)
        self.spatial_compression_ratio = arch.scale_factor_spatial
        self.temporal_compression_ratio = 1
        self.use_slicing = False
        self.use_tiling = config.use_tiling
        self.tile_sample_min_height = config.tile_sample_min_height
        self.tile_sample_min_width = config.tile_sample_min_width
        self.tile_sample_stride_height = config.tile_sample_stride_height
        self.tile_sample_stride_width = config.tile_sample_stride_width
        self._validate_tiling()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def _validate_tiling(self) -> None:
        ratio = self.spatial_compression_ratio
        for extent, stride in ((self.tile_sample_min_height, self.tile_sample_stride_height),
                               (self.tile_sample_min_width, self.tile_sample_stride_width)):
            if extent < ratio or stride < ratio or extent % ratio or stride % ratio or stride > extent:
                raise ValueError(
                    "VAE tile size and stride must be positive spatial-scale multiples, with stride <= size."
                )

    def enable_tiling(self, tile_sample_min_height: int | None = None, tile_sample_min_width: int | None = None,
                      tile_sample_stride_height: int | None = None, tile_sample_stride_width: int | None = None,
                      **kwargs) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self._validate_tiling()

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def clear_cache(self) -> None:
        """Image inference retains no temporal feature cache."""

    @staticmethod
    def _validate_image(x: torch.Tensor, channels: int, name: str) -> None:
        if x.ndim != 5 or x.shape[1] != channels or x.shape[2] != 1:
            raise ValueError(f"`{name}` must have shape [B, {channels}, 1, H, W], got {tuple(x.shape)}.")
        if x.shape[0] < 1 or min(x.shape[-2:]) < 1:
            raise ValueError(f"`{name}` must contain at least one nonempty image.")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.patch_size is not None:
            x = _patchify(x, self.config.patch_size)
        if self.use_tiling and (x.shape[-2] > self.tile_sample_min_height or x.shape[-1] > self.tile_sample_min_width):
            return self.tiled_encode(x)
        return self.quant_conv(self.encoder(x))

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        self._validate_image(x, self.config.in_channels, "x")
        if x.shape[-2] % self.spatial_compression_ratio or x.shape[-1] % self.spatial_compression_ratio:
            raise ValueError("Image height and width must be divisible by the VAE spatial compression ratio.")
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self._encode(item) for item in x.split(1)])
        else:
            moments = self._encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        return AutoencoderKLOutput(posterior) if return_dict else (posterior,)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        if self.use_tiling and (z.shape[-2] > min_height or z.shape[-1] > min_width):
            return self.tiled_decode(z).sample
        decoded = self.decoder(self.post_quant_conv(z))
        if self.config.patch_size is not None:
            decoded = _unpatchify(decoded, self.config.patch_size)
        return decoded.clamp(-1.0, 1.0)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        self._validate_image(z, self.z_dim, "z")
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(item) for item in z.split(1)])
        else:
            decoded = self._decode(z)
        return DecoderOutput(decoded) if return_dict else (decoded,)

    @staticmethod
    def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = (
                a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent)
                + b[:, :, :, y, :] * (y / blend_extent)
            )
        return b

    @staticmethod
    def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = (
                a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent)
                + b[:, :, :, :, x] * (x / blend_extent)
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        ratio = self.spatial_compression_ratio // (self.config.patch_size or 1)
        tile_height = self.tile_sample_min_height // ratio
        tile_width = self.tile_sample_min_width // ratio
        stride_height = self.tile_sample_stride_height // ratio
        stride_width = self.tile_sample_stride_width // ratio
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                tile = x[:, :, :, i:i + self.tile_sample_min_height, j:j + self.tile_sample_min_width]
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, tile_height - stride_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, tile_width - stride_width)
                result_row.append(tile[:, :, :, :stride_height, :stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=3)[:, :, :, :height // ratio, :width // ratio]

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        height, width = z.shape[-2:]
        ratio = self.spatial_compression_ratio
        tile_height = self.tile_sample_min_height // ratio
        tile_width = self.tile_sample_min_width // ratio
        stride_height = self.tile_sample_stride_height // ratio
        stride_width = self.tile_sample_stride_width // ratio
        sample_height, sample_width = height * ratio, width * ratio
        sample_stride_height = self.tile_sample_stride_height
        sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            patch_size = self.config.patch_size
            sample_height //= patch_size
            sample_width //= patch_size
            sample_stride_height //= patch_size
            sample_stride_width //= patch_size
            blend_height = self.tile_sample_min_height // patch_size - sample_stride_height
            blend_width = self.tile_sample_min_width // patch_size - sample_stride_width
        else:
            blend_height = self.tile_sample_min_height - sample_stride_height
            blend_width = self.tile_sample_min_width - sample_stride_width
        rows = []
        for i in range(0, height, stride_height):
            row = []
            for j in range(0, width, stride_width):
                tile = z[:, :, :, i:i + tile_height, j:j + tile_width]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :sample_stride_height, :sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))
        decoded = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
        if self.config.patch_size is not None:
            decoded = _unpatchify(decoded, self.config.patch_size)
        decoded = decoded.clamp(-1.0, 1.0)
        return DecoderOutput(decoded) if return_dict else (decoded,)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.config.latents_mean, device=latents.device, dtype=latents.dtype)
        std = torch.tensor(self.config.latents_std, device=latents.device, dtype=latents.dtype)
        mean, std = mean.view(1, self.z_dim, 1, 1, 1), std.view(1, self.z_dim, 1, 1, 1)
        return (latents - mean) / std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.config.latents_mean, device=latents.device, dtype=latents.dtype)
        std = torch.tensor(self.config.latents_std, device=latents.device, dtype=latents.dtype)
        mean, std = mean.view(1, self.z_dim, 1, 1, 1), std.view(1, self.z_dim, 1, 1, 1)
        return latents * std + mean

    def forward(self, sample: torch.Tensor, sample_posterior: bool = False, return_dict: bool = True,
                generator: torch.Generator | None = None) -> DecoderOutput | tuple[torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(latents, return_dict=return_dict)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, *, subfolder: str | None = None,
                        config: VAEConfig | None = None, torch_dtype: torch.dtype | None = None,
                        device: torch.device | str = "cpu", local_files_only: bool = True) -> AutoencoderKLQwenImage21:
        """Strictly load a local Diffusers-layout VAE component without downloading weights."""
        from safetensors.torch import load_file

        if not local_files_only:
            raise ValueError("Qwen-Image-2.1 VAE.from_pretrained accepts local checkpoints only.")
        path = Path(model_path)
        if subfolder:
            path /= subfolder
        if not path.is_dir():
            raise FileNotFoundError(f"Local Qwen-Image-2.1 VAE directory does not exist: {path}")
        with (path / "config.json").open() as handle:
            arch_fields = json.load(handle)
        class_name = arch_fields.get("_class_name")
        if class_name != "AutoencoderKLQwenImage21":
            raise ValueError(f"Expected AutoencoderKLQwenImage21 checkpoint, got {class_name!r}.")
        config = config or QwenImage21VAEConfig()
        config.update_model_arch(arch_fields)
        model = cls(config)
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors VAE checkpoint found in {path}.")
        state = {}
        for file in files:
            shard = load_file(str(file))
            duplicates = state.keys() & shard.keys()
            if duplicates:
                raise ValueError(f"Duplicate tensors in VAE checkpoint shards: {sorted(duplicates)[:5]}")
            state.update(shard)
        model.load_state_dict(state, strict=True)
        logger.info("Loaded Qwen-Image-2.1 VAE strictly from %s", path)
        return model.to(device=device, dtype=torch_dtype).eval()


EntryClass = AutoencoderKLQwenImage21
