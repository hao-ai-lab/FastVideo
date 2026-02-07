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

    def encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        """Encode to latent distribution."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        """Decode from latents."""
        z = self.post_quant_conv(z)
        dec = self.decoder(z, latent_embeds=None)
        return DecoderOutput(sample=dec)


EntryClass = GameCraftVAE
