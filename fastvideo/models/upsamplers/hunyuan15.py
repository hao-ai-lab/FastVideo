# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from fastvideo.models.vaes.hunyuan15vae import (
    HunyuanVideo15CausalConv3d,
    HunyuanVideo15RMS_norm,
)

from fastvideo.layers.activation import get_act_fn
from fastvideo.configs.models.upsamplers import SRTo720pUpsamplerConfig, SRTo1080pUpsamplerConfig

class HunyuanVideo15ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        non_linearity: str = "swish",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = get_act_fn(non_linearity)

        self.norm1 = HunyuanVideo15RMS_norm(in_channels, images=False)
        self.conv1 = HunyuanVideo15CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = HunyuanVideo15RMS_norm(out_channels, images=False)
        self.conv2 = HunyuanVideo15CausalConv3d(out_channels, out_channels, kernel_size=3)

        self.nin_shortcut = None
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.nin_shortcut is not None:
            residual = self.nin_shortcut(residual)

        return hidden_states + residual

class SRResidualCausalBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            HunyuanVideo15CausalConv3d(channels, channels, kernel_size=3),
            nn.SiLU(inplace=True),
            HunyuanVideo15CausalConv3d(channels, channels, kernel_size=3),
            nn.SiLU(inplace=True),
            HunyuanVideo15CausalConv3d(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SRTo720pUpsampler(nn.Module):

    def __init__(
        self,
        config: SRTo720pUpsamplerConfig,
    ):
        super().__init__()
        self.in_conv = HunyuanVideo15CausalConv3d(config.in_channels, config.hidden_channels, kernel_size=3)
        self.blocks = nn.ModuleList([SRResidualCausalBlock3D(config.hidden_channels) for _ in range(config.num_blocks)])
        self.out_conv = HunyuanVideo15CausalConv3d(config.hidden_channels, config.out_channels, kernel_size=3)
        self.global_residual = bool(config.global_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.in_conv(x)
        for blk in self.blocks:
            y = blk(y)
        y = self.out_conv(y)
        if self.global_residual and (y.shape == residual.shape):
            y = y + residual
        return y


class SRTo1080pUpsampler(nn.Module):

    def __init__(
        self,
        config: SRTo1080pUpsamplerConfig,
    ):
        super().__init__()
        self.num_res_blocks = config.num_res_blocks
        self.block_out_channels = config.block_out_channels
        self.z_channels = config.z_channels

        block_in = config.block_out_channels[0]
        self.conv_in = HunyuanVideo15CausalConv3d(config.z_channels, block_in, kernel_size=3)

        self.up = nn.ModuleList()
        for i_level, ch in enumerate(config.block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(HunyuanVideo15ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            self.up.append(up)

        self.norm_out = HunyuanVideo15RMS_norm(block_in, images=False)
        self.conv_out = HunyuanVideo15CausalConv3d(block_in, config.out_channels, kernel_size=3)

        self.gradient_checkpointing = False
        self.is_residual = config.is_residual

    def forward(self, z: Tensor, target_shape: Sequence[int] = None) -> Tensor:
        """
        Args:
            z: (B, C, T, H, W)
            target_shape: (H, W)
        """
        if target_shape is not None and z.shape[-2:] != target_shape:
            bsz = z.shape[0]
            z = rearrange(z, "b c f h w -> (b f) c h w")
            z = F.interpolate(z, size=target_shape, mode="bilinear", align_corners=False)
            z = rearrange(z, "(b f) c h w -> b c f h w", b=bsz)

        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = get_act_fn("swish")(h)
        h = self.conv_out(h)
        return h