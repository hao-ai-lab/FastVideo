# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image VAE: the video causal encoder plus a 2D ViT decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from fastvideo.configs.models.vaes.minimax_h3_image import MiniMaxH3ImageVAEConfig
from fastvideo.models.vaes.minimax_h3_video import (
    AutoencoderKLOutput,
    DecoderOutput,
    DiagonalGaussianDistribution,
    MiniMaxH3VideoEncoder3d,
    MiniMaxH3VideoRotaryPosEmbed,
    MiniMaxH3VideoTransformerBlock,
)

_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)


class MiniMaxH3ImageViTDecoder2d(nn.Module):
    """Non-causal 2D decoder. Each latent pixel is one token; registers are dropped before the patch projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
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
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rope = MiniMaxH3VideoRotaryPosEmbed(
            int(attention_head_dim * rope_dim_ratio),
            theta=rope_theta,
            num_axes=2,
        )
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList([
            MiniMaxH3VideoTransformerBlock(
                dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                ffn_mult=ffn_mult,
                eps=norm_eps,
            ) for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size * patch_size)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, num_channels)
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 2))
        rotary_emb = self.rope(torch.cat([position_ids, suffix_ids], dim=1))

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = checkpoint(block, hidden_states, rotary_emb, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.proj_out(self.norm_out(hidden_states).to(self.proj_out.weight.dtype))
        hidden_states = hidden_states[:, :num_patches, :]
        patch_size = self.patch_size
        hidden_states = hidden_states.view(batch_size, height, width, self.out_channels, patch_size, patch_size)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5).contiguous()
        return hidden_states.reshape(batch_size, self.out_channels, height * patch_size, width * patch_size)


class AutoencoderKLMiniMaxH3Image(nn.Module):
    """Causal 3D encoder over one frame, then a 2D ViT decoder.

    Pixels are ImageNet-normalized RGB on ``[0, 1]``. Latents use the published
    per-channel mean and std. Tiling is on by default because the release
    images are the blended-tile ones. Weights stay in FP32.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3VideoResnetBlock3d", "MiniMaxH3VideoTransformerBlock"]
    _keep_in_fp32_modules = ["encoder", "decoder", "quant_conv", "post_quant_conv"]

    def __init__(self, config: MiniMaxH3ImageVAEConfig) -> None:
        super().__init__()
        self.model_config = config
        self.config = config.arch_config
        arch = config.arch_config
        self.latent_channels = int(arch.latent_channels)
        self.spatial_compression_ratio = math.prod(arch.spatial_downsample_factors)

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
        self.post_quant_conv = nn.Conv2d(arch.latent_channels, arch.latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3ImageViTDecoder2d(
            in_channels=arch.latent_channels,
            out_channels=arch.out_channels,
            patch_size=self.spatial_compression_ratio,
            num_layers=arch.decoder_num_layers,
            num_attention_heads=arch.decoder_num_attention_heads,
            attention_head_dim=arch.decoder_attention_head_dim,
            num_register_tokens=arch.decoder_num_register_tokens,
            ffn_mult=arch.decoder_ffn_mult,
            rope_theta=arch.decoder_rope_theta,
            rope_dim_ratio=arch.decoder_rope_dim_ratio,
            norm_eps=arch.decoder_norm_eps,
        )

        self.use_slicing = False
        self.use_tiling = config.use_tiling
        self.tile_sample_min_height = config.tile_sample_min_height
        self.tile_sample_min_width = config.tile_sample_min_width
        self.tile_sample_min_overlap_height = config.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = config.tile_sample_min_overlap_width

        self.register_buffer(
            "latents_mean",
            torch.tensor(arch.latents_mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(arch.latents_std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor(_PIXEL_MEAN, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(_PIXEL_STD, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.float()

    def to(self, *args, **kwargs) -> "AutoencoderKLMiniMaxH3Image":
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None or dtype == torch.float32:
            return super().to(*args, **kwargs)
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            return super().to(*args, **kwargs)
        to_kwargs = {"device": device, "dtype": torch.float32, "non_blocking": non_blocking}
        if memory_format is not None:
            to_kwargs["memory_format"] = memory_format
        return super().to(**to_kwargs)

    def half(self) -> "AutoencoderKLMiniMaxH3Image":
        return self

    def bfloat16(self) -> "AutoencoderKLMiniMaxH3Image":
        return self

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latents_mean) / self.latents_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latents_std + self.latents_mean

    def normalize_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        return (pixels - self.pixel_mean) / self.pixel_std

    def denormalize_pixels(self, sample: torch.Tensor) -> torch.Tensor:
        return sample * self.pixel_std + self.pixel_mean

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
        starts = [0]
        for index in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[index])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(first: torch.Tensor, second: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(first.shape[dim], second.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=second.device, dtype=second.dtype)
        shape = [1] * first.ndim
        shape[dim] = blend_extent
        weight_first = (1 - positions / blend_extent).view(shape)
        weight_second = (positions / blend_extent).view(shape)
        slice_first = [slice(None)] * first.ndim
        slice_first[dim] = slice(-blend_extent, None)
        slice_second = [slice(None)] * second.ndim
        slice_second[dim] = slice(0, blend_extent)
        blended = first[tuple(slice_first)] * weight_first + second[tuple(slice_second)] * weight_second
        if blend_extent == second.shape[dim]:
            return blended
        rest = [slice(None)] * second.ndim
        rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, second[tuple(rest)]], dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for row_index, row in enumerate(tiles):
            result_row = []
            for col_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend(tiles[row_index - 1][col_index], tile, height_overlaps[row_index - 1], dim=-2)
                if col_index > 0:
                    tile = self._blend(row[col_index - 1], tile, width_overlaps[col_index - 1], dim=-1)
                if row_index < len(tiles) - 1:
                    tile = tile[..., :-height_overlaps[row_index], :]
                if col_index < len(row) - 1:
                    tile = tile[..., :, :-width_overlaps[col_index]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.quant_conv(self.encoder(pixels[:, :, None]))[:, :, -1]
        height, width = pixels.shape[-2], pixels.shape[-1]
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
        for y_pos, y_len in zip(y_indices, y_lengths, strict=True):
            row = []
            for x_pos, x_len in zip(x_indices, x_lengths, strict=True):
                tile = pixels[..., y_pos:y_pos + y_len, x_pos:x_pos + x_len]
                row.append(self.quant_conv(self.encoder(tile[:, :, None]))[:, :, -1])
            rows.append(row)
        ratio = self.spatial_compression_ratio
        return self._stitch_tiles(
            rows,
            [overlap // ratio for overlap in y_overlaps],
            [overlap // ratio for overlap in x_overlaps],
        )

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(latents))
        ratio = self.spatial_compression_ratio
        height = latents.shape[-2] * ratio
        width = latents.shape[-1] * ratio
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
        for y_pos, y_len in zip(y_indices, y_lengths, strict=True):
            row = []
            for x_pos, x_len in zip(x_indices, x_lengths, strict=True):
                tile = latents[..., y_pos // ratio:y_pos // ratio + y_len // ratio,
                               x_pos // ratio:x_pos // ratio + x_len // ratio]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def encode(self, pixels: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple:
        pixels = pixels.to(dtype=torch.float32)
        if self.use_slicing and pixels.shape[0] > 1:
            moments = torch.cat([self._encode(chunk) for chunk in pixels.split(1)])
        else:
            moments = self._encode(pixels)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior, )
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, latents: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple:
        latents = latents.to(dtype=torch.float32)
        if self.use_slicing and latents.shape[0] > 1:
            decoded = torch.cat([self._decode(chunk) for chunk in latents.split(1)])
        else:
            decoded = self._decode(latents)
        if not return_dict:
            return (decoded, )
        return DecoderOutput(sample=decoded)


EntryClass = AutoencoderKLMiniMaxH3Image
