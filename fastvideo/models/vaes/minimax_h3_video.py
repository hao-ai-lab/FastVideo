# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native MiniMax-H3 video VAE.

The encoder is a causal 3D CNN while the decoder is a full-attention ViT.
This module intentionally uses only PyTorch and FastVideo configuration types.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior used by the KL encoder."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise_device = self.mean.device
        if generator is not None and generator.device.type == "cpu":
            noise_device = torch.device("cpu")
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=noise_device,
            dtype=self.mean.dtype,
        ).to(self.mean.device)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


@dataclass
class AutoencoderKLOutput:
    latent_dist: DiagonalGaussianDistribution


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    """3D convolution with reflect spatial padding and causal temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states,
                (padding, padding, padding, padding, 0, 0),
                mode=self.spatial_padding_mode,
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
    """GroupNorm with each temporal frame normalized independently."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for index in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3VideoDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = checkpoint(resnet, hidden_states, use_reentrant=False)
            else:
                hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        spatial_downsample_factors: tuple[int, ...],
        temporal_downsample_factors: tuple[int, ...],
        norm_num_groups: int,
        norm_eps: float,
        spatial_padding_mode: str,
    ) -> None:
        super().__init__()
        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[index],
                    out_channels=block_out_channels[index],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[index],
                    spatial_downsample_factor=spatial_downsample_factors[index],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for index in range(len(block_out_channels))
            ]
        )
        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        return self.conv_out(F.silu(self.norm_out(hidden_states)))


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"`dim` {dim} must be divisible by `2 * num_axes` {2 * num_axes}.")
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class MiniMaxH3VideoAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.use_bias = bias
        inner_dim = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(0.0)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(2, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(2, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(2, (self.heads, -1))

        query = self.norm_q(query.float()).to(query.dtype)
        key = self.norm_k(key.float()).to(key.dtype)
        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            rotary_dim = cos.shape[-1]
            query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
            key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
            query_first, query_second = query_rotary.chunk(2, dim=-1)
            key_first, key_second = key_rotary.chunk(2, dim=-1)
            query_rotated = torch.cat([-query_second, query_first], dim=-1)
            key_rotated = torch.cat([-key_second, key_first], dim=-1)
            query = torch.cat([query_rotary * cos + query_rotated * sin, query_pass], dim=-1)
            key = torch.cat([key_rotary * cos + key_rotated * sin, key_pass], dim=-1)

        query, key, value = (tensor.permute(0, 2, 1, 3) for tensor in (query, key, value))
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.permute(0, 2, 1, 3).flatten(2, 3)
        return self.to_out[0](hidden_states)


class MiniMaxH3VideoSwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class MiniMaxH3VideoFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, bias: bool = True) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList(
            [
                MiniMaxH3VideoSwiGLU(dim, inner_dim, bias=bias),
                nn.Dropout(0.0),
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3VideoAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = MiniMaxH3VideoFeedForward(dim, mult=ffn_mult, bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(norm_hidden_states, rotary_emb) * self.scale1
        norm_hidden_states = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        return hidden_states + self.ff(norm_hidden_states) * self.scale2


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        patch_size_t: int,
        num_layers: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_register_tokens: int,
        ffn_mult: int,
        rope_theta: float,
        rope_dim_ratio: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size,
            num_frames * height * width,
            num_channels,
        )
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        rotary_emb = self.rope(torch.cat([position_ids, suffix_ids], dim=1))

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = checkpoint(block, hidden_states, rotary_emb, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :num_patches, :]
        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


class AutoencoderKLMiniMaxH3(nn.Module):
    """MiniMax-H3 causal encoder and ViT decoder with exact release geometry."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3VideoResnetBlock3d", "MiniMaxH3VideoTransformerBlock"]
    _repeated_blocks = ["MiniMaxH3VideoTransformerBlock"]
    _keep_in_fp32_modules = ["encoder", "decoder", "quant_conv", "post_quant_conv"]

    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        super().__init__()
        self.model_config = config
        self.config = config.arch_config
        arch = config.arch_config
        self.latent_channels = int(arch.latent_channels)
        self.spatial_compression_ratio = math.prod(arch.spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(arch.temporal_downsample_factors)

        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=arch.in_channels,
            out_channels=2 * arch.latent_channels,
            block_out_channels=tuple(arch.block_out_channels),
            layers_per_block=arch.layers_per_block,
            spatial_downsample_factors=tuple(arch.spatial_downsample_factors),
            temporal_downsample_factors=tuple(arch.temporal_downsample_factors),
            norm_num_groups=arch.norm_num_groups,
            norm_eps=arch.norm_eps,
            spatial_padding_mode=arch.spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * arch.latent_channels, 2 * arch.latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(arch.latent_channels, arch.latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=arch.latent_channels,
            out_channels=arch.out_channels,
            patch_size=self.spatial_compression_ratio,
            patch_size_t=self.temporal_compression_ratio,
            num_layers=arch.decoder_num_layers,
            num_attention_heads=arch.decoder_num_attention_heads,
            attention_head_dim=arch.decoder_attention_head_dim,
            num_register_tokens=arch.decoder_num_register_tokens,
            ffn_mult=arch.decoder_ffn_mult,
            rope_theta=arch.decoder_rope_theta,
            rope_dim_ratio=arch.decoder_rope_dim_ratio,
            norm_eps=arch.decoder_norm_eps,
        )

        self.frame_pre_padding = (-arch.clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(arch.clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-arch.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(
            self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding,
            0,
        )
        self.use_slicing = False
        self.use_tiling = config.use_tiling
        self.tile_sample_min_height = config.tile_sample_min_height
        self.tile_sample_min_width = config.tile_sample_min_width
        self.tile_sample_min_overlap_height = config.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = config.tile_sample_min_overlap_width

        self.register_buffer(
            "latents_mean",
            torch.tensor(arch.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(arch.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        # The released encoder and decoder stay in FP32 for both weights and compute.
        self.float()

    def to(self, *args, **kwargs) -> "AutoencoderKLMiniMaxH3":
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None or dtype == torch.float32:
            return super().to(*args, **kwargs)
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            return super().to(*args, **kwargs)

        to_kwargs = {
            "device": device,
            "dtype": torch.float32,
            "non_blocking": non_blocking,
        }
        if memory_format is not None:
            to_kwargs["memory_format"] = memory_format
        return super().to(**to_kwargs)

    def half(self) -> "AutoencoderKLMiniMaxH3":
        return self

    def bfloat16(self) -> "AutoencoderKLMiniMaxH3":
        return self

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latents_mean) / self.latents_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latents_std + self.latents_mean

    def normalize_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        return (pixels - self.pixel_mean) / self.pixel_std

    def denormalize_pixels(self, sample: torch.Tensor) -> torch.Tensor:
        return sample * self.pixel_std + self.pixel_mean

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = tile_sample_min_overlap_width or self.tile_sample_min_overlap_width

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.spatial_compression_ratio):
            overlaps[index % (num_tiles - 1)] += self.spatial_compression_ratio
        tile_start_indices = [0]
        for index in range(num_tiles - 1):
            tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[index])
        return tile_start_indices, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for row_index, row in enumerate(tiles):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend(tiles[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2)
                if column_index > 0:
                    tile = self._blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
                if row_index < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[column_index]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode_clip(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x))
        height, width = x.shape[-2:]
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height,
            self.tile_sample_min_height,
            self.tile_sample_min_overlap_height,
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width,
            self.tile_sample_min_width,
            self.tile_sample_min_overlap_width,
        )
        rows = []
        for y_position, y_length in zip(y_indices, y_lengths):
            row = []
            for x_position, x_length in zip(x_indices, x_lengths):
                tile = x[..., y_position:y_position + y_length, x_position:x_position + x_length]
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)
        latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height,
            self.tile_sample_min_height,
            self.tile_sample_min_overlap_height,
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width,
            self.tile_sample_min_width,
            self.tile_sample_min_overlap_width,
        )
        ratio = self.spatial_compression_ratio
        rows = []
        for y_position, y_length in zip(y_indices, y_lengths):
            row = []
            for x_position, x_length in zip(x_indices, x_lengths):
                tile = z[
                    ...,
                    y_position // ratio:y_position // ratio + y_length // ratio,
                    x_position // ratio:x_position // ratio + x_length // ratio,
                ]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        clip_length = self.config.clip_length
        num_frames = x.shape[2]
        if num_frames % clip_length != 0:
            pad_frames = x[:, :, -1:].repeat(1, 1, (-num_frames) % clip_length, 1, 1)
            x = torch.cat([x, pad_frames], dim=2)
        moments = torch.cat(
            [
                self._encode_clip(x[:, :, index * clip_length:(index + 1) * clip_length])
                for index in range(x.shape[2] // clip_length)
            ],
            dim=2,
        )
        if self.config.token_drop > 0:
            moments = moments[:, :, :-self.config.token_drop]
        return moments

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio
        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        decoded_chunks = []
        overlap = None
        for index in range(num_chunks):
            start = index * tokens_chunk_size
            clip = self._decode_clip(z[:, :, start:start + tokens_chunk_size + self.token_overlap])
            for overlap_index in range(int(token_drop > 0) + 1):
                frame_start = overlap_index * chunk_num_frames
                chunk = clip[:, :, frame_start:frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding:]
                if overlap_index == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)
        decoded = torch.cat(decoded_chunks, dim=2)

        if pad_tokens > 0:
            intra_tail = self.config.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail
                if intra_tail and (num_tokens_before_pad + offset) % tokens_chunk_size == 0
                else temporal_ratio
                for offset in range(pad_tokens)
            )
            decoded = decoded[:, :, :-pad_frames]
        return decoded

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self._encode(x_slice) for x_slice in x.split(1)])
        else:
            moments = self._encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def encode_keyframe(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        """Encode one-frame conditioning inputs without video chunk padding."""
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError(f"`x` must contain exactly one video frame, got shape {tuple(x.shape)}.")
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self._encode_clip(x_slice) for x_slice in x.split(1)])
        else:
            moments = self._encode_clip(x)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)])
        else:
            decoded = self._decode(z)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(latents, return_dict=return_dict)


EntryClass = AutoencoderKLMiniMaxH3
