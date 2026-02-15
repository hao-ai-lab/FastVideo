# SPDX-License-Identifier: Apache-2.0
"""
GameCraft VAE - ported from official Hunyuan-GameCraft-1.0/hymm_sp/vae/.

Matches the official AutoencoderKLCausal3D structure exactly for weight loading.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fastvideo.configs.models.vaes.gamecraftvae import GameCraftVAEConfig
from fastvideo.models.vaes.common import DiagonalGaussianDistribution
from fastvideo.models.vaes.gamecraftvae_blocks import (
    CausalConv3d,
    DownEncoderBlockCausal3D,
    UpDecoderBlockCausal3D,
    UNetMidBlockCausal3D,
)


@dataclass
class AutoencoderKLOutput:
    """Matches official AutoencoderKLOutput interface."""

    latent_dist: DiagonalGaussianDistribution


@dataclass
class DecoderOutput:
    """Matches official DecoderOutput interface."""

    sample: torch.Tensor


class EncoderCausal3D(nn.Module):
    """Encoder - ported from official vae.py. Structure matches for weight loading."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlockCausal3D",),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        disable_causal: bool = False,
        mid_block_causal_attn: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, disable_causal=disable_causal
        )
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, _ in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial = int(np.log2(spatial_compression_ratio))
            num_time = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial = bool(i < num_spatial)
                add_time = bool(
                    i >= (len(block_out_channels) - 1 - num_time) and not is_final_block
                )
            elif time_compression_ratio == 8:
                add_spatial = bool(i < num_spatial)
                add_time = bool(i < num_time)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")

            downsample_stride_HW = (2, 2) if add_spatial else (1, 1)
            downsample_stride_T = (2,) if add_time else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)

            down_block = DownEncoderBlockCausal3D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                add_downsample=bool(add_spatial or add_time),
                downsample_stride=downsample_stride,
                downsample_padding=0,
                disable_causal=disable_causal,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            num_layers=1,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            disable_causal=disable_causal,
            causal_attention=mid_block_causal_attn,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(
            block_out_channels[-1],
            conv_out_channels,
            kernel_size=3,
            disable_causal=disable_causal,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample, scale=1.0)
        sample = self.mid_block(sample, temb=None)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class DecoderCausal3D(nn.Module):
    """Decoder - ported from official vae.py. Structure matches for weight loading."""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlockCausal3D",),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        disable_causal: bool = False,
        mid_block_causal_attn: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            disable_causal=disable_causal,
        )

        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            num_layers=1,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            disable_causal=disable_causal,
            causal_attention=mid_block_causal_attn,
        )

        self.up_blocks = nn.ModuleList([])
        reversed_channels = list(reversed(block_out_channels))
        output_channel = reversed_channels[0]
        for i, _ in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial = int(np.log2(spatial_compression_ratio))
            num_time = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial = bool(i < num_spatial)
                add_time = bool(
                    i >= len(block_out_channels) - 1 - num_time and not is_final_block
                )
            elif time_compression_ratio == 8:
                add_spatial = bool(i >= len(block_out_channels) - num_spatial)
                add_time = bool(i >= len(block_out_channels) - num_time)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")

            upsample_HW = (2, 2) if add_spatial else (1, 1)
            upsample_T = (2,) if add_time else (1,)
            upsample_scale_factor = tuple(upsample_T + upsample_HW)

            up_block = UpDecoderBlockCausal3D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=layers_per_block + 1,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                add_upsample=bool(add_spatial or add_time),
                upsample_scale_factor=upsample_scale_factor,
                temb_channels=None,
                disable_causal=disable_causal,
            )
            self.up_blocks.append(up_block)
            output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(
            block_out_channels[0], out_channels, kernel_size=3, disable_causal=disable_causal
        )

    def forward(
        self,
        sample: torch.Tensor,
        latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, temb=latent_embeds)
        for up_block in self.up_blocks:
            sample = up_block(sample, temb=latent_embeds, scale=1.0)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class GameCraftVAE(nn.Module):
    """
    GameCraft VAE - ported from official AutoencoderKLCausal3D.
    Structure matches exactly for loading official weights.
    """

    def __init__(self, config: GameCraftVAEConfig):
        super().__init__()
        self.config = config
        arch = config.arch_config

        time_ratio = getattr(arch, "time_compression_ratio", arch.temporal_compression_ratio)
        self.encoder = EncoderCausal3D(
            in_channels=arch.in_channels,
            out_channels=arch.latent_channels,
            down_block_types=tuple(arch.down_block_types),
            block_out_channels=tuple(arch.block_out_channels),
            layers_per_block=arch.layers_per_block,
            norm_num_groups=arch.norm_num_groups,
            act_fn=arch.act_fn,
            double_z=True,
            time_compression_ratio=time_ratio,
            spatial_compression_ratio=arch.spatial_compression_ratio,
            disable_causal=getattr(arch, "disable_causal_conv", False),
            mid_block_add_attention=arch.mid_block_add_attention,
            mid_block_causal_attn=getattr(arch, "mid_block_causal_attn", False),
        )

        self.decoder = DecoderCausal3D(
            in_channels=arch.latent_channels,
            out_channels=arch.out_channels,
            up_block_types=tuple(arch.up_block_types),
            block_out_channels=tuple(arch.block_out_channels),
            layers_per_block=arch.layers_per_block,
            norm_num_groups=arch.norm_num_groups,
            act_fn=arch.act_fn,
            time_compression_ratio=time_ratio,
            spatial_compression_ratio=arch.spatial_compression_ratio,
            disable_causal=getattr(arch, "disable_causal_conv", False),
            mid_block_add_attention=arch.mid_block_add_attention,
            mid_block_causal_attn=getattr(arch, "mid_block_causal_attn", False),
        )

        self.quant_conv = nn.Conv3d(
            2 * arch.latent_channels, 2 * arch.latent_channels, kernel_size=1
        )
        self.post_quant_conv = nn.Conv3d(
            arch.latent_channels, arch.latent_channels, kernel_size=1
        )
        
        # Scaling factor for latent normalization (required for decoding stage)
        self.scaling_factor = arch.scaling_factor
        
        # Tiling support - matches official GameCraft VAE settings
        self._tiling_enabled = False
        self.tile_overlap_factor = 0.25
        
        # Temporal tiling params (for >64 output frames)
        self.tile_sample_min_tsize = 64  # Minimum sample temporal size (video frames)
        self.tile_latent_min_tsize = 16  # = 64 // 4 (time_compression_ratio)
        
        # Spatial tiling params - use small tiles to reduce memory
        self.tile_sample_min_size = 256  # Minimum spatial tile size in pixel space
        self.tile_latent_min_size = 32   # = 256 // 8 (spatial_compression_ratio)

    def encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        """Encode to latent distribution."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latents.
        
        Args:
            z: Latent tensor [B, C, T, H, W]
            
        Returns:
            Decoded tensor [B, C, T_out, H_out, W_out]
        """
        # Use tiled decode for memory efficiency when enabled
        if self._tiling_enabled:
            # Check if temporal tiling needed (>64 output frames)
            if z.shape[2] > self.tile_latent_min_tsize:
                return self._temporal_tiled_decode(z)
            # Check if spatial tiling needed (large H or W)
            if z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size:
                return self._spatial_tiled_decode(z)
        
        z = self.post_quant_conv(z)
        dec = self.decoder(z, latent_embeds=None)
        return dec

    def _temporal_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents in temporal tiles with overlapping and blending.
        
        Based on official GameCraft temporal_tiled_decode implementation.
        Only used when T > tile_latent_min_tsize (16).
        """
        B, C, T, H, W = z.shape
        
        # Use the pre-configured tiling parameters
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent
        
        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
            tile = self.post_quant_conv(tile)
            decoded = self.decoder(tile, latent_embeds=None)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]  # Skip first frame for non-first tiles
            row.append(decoded)
        
        # Blend overlapping regions
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self._blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, :t_limit+1, :, :])
        
        return torch.cat(result_row, dim=2)
    
    def _spatial_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents in spatial tiles with overlapping and blending.
        
        Based on official GameCraft spatial_tiled_decode implementation.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
        
        # Split z into overlapping tiles and decode them separately
        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[:, :, :, i: i + self.tile_latent_min_size, j: j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile, latent_embeds=None)
                row.append(decoded)
            rows.append(row)
        
        # Blend overlapping regions
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # Blend with above tile and left tile
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        
        return torch.cat(result_rows, dim=-2)
    
    def _blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend two tensors along temporal dimension."""
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        if blend_extent == 0:
            return b
        
        a_region = a[..., -blend_extent:, :, :]
        b_region = b[..., :blend_extent, :, :]
        
        weights = torch.arange(blend_extent, device=a.device, dtype=a.dtype) / blend_extent
        weights = weights.view(1, 1, blend_extent, 1, 1)
        
        blended = a_region * (1 - weights) + b_region * weights
        b[..., :blend_extent, :, :] = blended
        return b
    
    def _blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend two tensors along vertical (height) dimension."""
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        if blend_extent == 0:
            return b
        
        a_region = a[..., -blend_extent:, :]
        b_region = b[..., :blend_extent, :]
        
        weights = torch.arange(blend_extent, device=a.device, dtype=a.dtype) / blend_extent
        weights = weights.view(1, 1, 1, blend_extent, 1)
        
        blended = a_region * (1 - weights) + b_region * weights
        b[..., :blend_extent, :] = blended
        return b
    
    def _blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend two tensors along horizontal (width) dimension."""
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        if blend_extent == 0:
            return b
        
        a_region = a[..., -blend_extent:]
        b_region = b[..., :blend_extent]
        
        weights = torch.arange(blend_extent, device=a.device, dtype=a.dtype) / blend_extent
        weights = weights.view(1, 1, 1, 1, blend_extent)
        
        blended = a_region * (1 - weights) + b_region * weights
        b[..., :blend_extent] = blended
        return b

    def enable_tiling(self) -> None:
        """Enable tiling for large inputs."""
        self._tiling_enabled = True

    def disable_tiling(self) -> None:
        """Disable tiling."""
        self._tiling_enabled = False


EntryClass = GameCraftVAE
