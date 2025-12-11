# SPDX-License-Identifier: Apache-2.0
# Adapted from diffusers

# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from fastvideo.layers.activation import get_act_fn
from fastvideo.configs.models.vaes import Hunyuan15VAEConfig
from fastvideo.models.vaes.common import ParallelTiledVAE

class HunyuanVideo15CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.pad_mode = pad_mode
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(hidden_states)


class HunyuanVideo15RMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class HunyuanVideo15AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = HunyuanVideo15RMS_norm(in_channels, images=False)

        self.to_q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.to_k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.to_v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    @staticmethod
    def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
        """Prepare a causal attention mask for 3D videos.

        Args:
            n_frame (int): Number of frames (temporal length).
            n_hw (int): Product of height and width.
            dtype: Desired mask dtype.
            device: Device for the mask.
            batch_size (int, optional): If set, expands for batch.

        Returns:
            torch.Tensor: Causal attention mask.
        """
        seq_len = n_frame * n_hw
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        for i in range(seq_len):
            i_frame = i // n_hw
            mask[i, : (i_frame + 1) * n_hw] = 0
        if batch_size is not None:
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.norm(x)

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        batch_size, channels, frames, height, width = query.shape

        query = query.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        key = key.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        value = value.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()

        attention_mask = self.prepare_causal_attention_mask(
            frames, height * width, query.dtype, query.device, batch_size=batch_size
        )

        x = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        # batch_size, 1, frames * height * width, channels

        x = x.squeeze(1).reshape(batch_size, frames, height, width, channels).permute(0, 4, 1, 2, 3)
        x = self.proj_out(x)

        return x + identity


class HunyuanVideo15Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = HunyuanVideo15CausalConv3d(in_channels, out_channels * factor, kernel_size=3)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    @staticmethod
    def _dcae_upsample_rearrange(tensor, r1=1, r2=2, r3=2):
        """
        Convert (b, r1*r2*r3*c, f, h, w) -> (b, c, r1*f, r2*h, r3*w)

        Args:
            tensor: Input tensor of shape (b, r1*r2*r3*c, f, h, w)
            r1: temporal upsampling factor
            r2: height upsampling factor
            r3: width upsampling factor
        """
        b, packed_c, f, h, w = tensor.shape
        factor = r1 * r2 * r3
        c = packed_c // factor

        tensor = tensor.view(b, r1, r2, r3, c, f, h, w)
        tensor = tensor.permute(0, 4, 5, 1, 6, 2, 7, 3)
        return tensor.reshape(b, c, f * r1, h * r2, w * r3)

    def forward(self, x: torch.Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h_first = h[:, :, :1, :, :]
            h_first = self._dcae_upsample_rearrange(h_first, r1=1, r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            h_next = h[:, :, 1:, :, :]
            h_next = self._dcae_upsample_rearrange(h_next, r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = self._dcae_upsample_rearrange(x_first, r1=1, r2=2, r3=2)
            x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)

            x_next = x[:, :, 1:, :, :]
            x_next = self._dcae_upsample_rearrange(x_next, r1=r1, r2=2, r3=2)
            x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = torch.cat([x_first, x_next], dim=2)

        else:
            h = self._dcae_upsample_rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = self._dcae_upsample_rearrange(shortcut, r1=r1, r2=2, r3=2)
        return h + shortcut


class HunyuanVideo15Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        self.conv = HunyuanVideo15CausalConv3d(in_channels, out_channels // factor, kernel_size=3)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    @staticmethod
    def _dcae_downsample_rearrange(tensor, r1=1, r2=2, r3=2):
        """
        Convert (b, c, r1*f, r2*h, r3*w) -> (b, r1*r2*r3*c, f, h, w)

        This packs spatial/temporal dimensions into channels (opposite of upsample)
        """
        b, c, packed_f, packed_h, packed_w = tensor.shape
        f, h, w = packed_f // r1, packed_h // r2, packed_w // r3

        tensor = tensor.view(b, c, f, r1, h, r2, w, r3)
        tensor = tensor.permute(0, 3, 5, 7, 1, 2, 4, 6)
        return tensor.reshape(b, r1 * r2 * r3 * c, f, h, w)

    def forward(self, x: torch.Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        if self.add_temporal_downsample:
            h_first = h[:, :, :1, :, :]
            h_first = self._dcae_downsample_rearrange(h_first, r1=1, r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:, :, :]
            h_next = self._dcae_downsample_rearrange(h_next, r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = self._dcae_downsample_rearrange(x_first, r1=1, r2=2, r3=2)
            B, C, T, H, W = x_first.shape
            x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)
            x_next = x[:, :, 1:, :, :]
            x_next = self._dcae_downsample_rearrange(x_next, r1=r1, r2=2, r3=2)
            B, C, T, H, W = x_next.shape
            x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = self._dcae_downsample_rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = self._dcae_downsample_rearrange(x, r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)

        return h + shortcut


class HunyuanVideo15ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        non_linearity: str = "swish",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = get_act_fn(non_linearity)

        self.norm1 = HunyuanVideo15RMS_norm(in_channels, images=False)
        self.conv1 = HunyuanVideo15CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = HunyuanVideo15RMS_norm(out_channels, images=False)
        self.conv2 = HunyuanVideo15CausalConv3d(out_channels, out_channels, kernel_size=3)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


class HunyuanVideo15MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        add_attention: bool = True,
    ) -> None:
        super().__init__()
        self.add_attention = add_attention

        # There is always at least one resnet
        resnets = [
            HunyuanVideo15ResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(HunyuanVideo15AttnBlock(in_channels))
            else:
                attentions.append(None)

            resnets.append(
                HunyuanVideo15ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)

        return hidden_states


class HunyuanVideo15DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample_out_channels: Optional[int] = None,
        add_temporal_downsample: int = True,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                HunyuanVideo15ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if downsample_out_channels is not None:
            self.downsamplers = nn.ModuleList(
                [
                    HunyuanVideo15Downsample(
                        out_channels,
                        out_channels=downsample_out_channels,
                        add_temporal_downsample=add_temporal_downsample,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class HunyuanVideo15UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample_out_channels: Optional[int] = None,
        add_temporal_upsample: bool = True,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                HunyuanVideo15ResnetBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if upsample_out_channels is not None:
            self.upsamplers = nn.ModuleList(
                [
                    HunyuanVideo15Upsample(
                        out_channels,
                        out_channels=upsample_out_channels,
                        add_temporal_upsample=add_temporal_upsample,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for resnet in self.resnets:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)

        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class HunyuanVideo15Encoder3D(nn.Module):
    r"""
    3D vae encoder for HunyuanImageRefiner.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024, 1024),
        layers_per_block: int = 2,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 16,
        downsample_match_channel: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_size = block_out_channels[-1] // self.out_channels

        self.conv_in = HunyuanVideo15CausalConv3d(in_channels, block_out_channels[0], kernel_size=3)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        input_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            add_spatial_downsample = i < np.log2(spatial_compression_ratio)
            output_channel = block_out_channels[i]
            if not add_spatial_downsample:
                down_block = HunyuanVideo15DownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=None,
                    add_temporal_downsample=False,
                )
                input_channel = output_channel
            else:
                add_temporal_downsample = i >= np.log2(spatial_compression_ratio // temporal_compression_ratio)
                downsample_out_channels = block_out_channels[i + 1] if downsample_match_channel else output_channel
                down_block = HunyuanVideo15DownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=downsample_out_channels,
                    add_temporal_downsample=add_temporal_downsample,
                )
                input_channel = downsample_out_channels

            self.down_blocks.append(down_block)

        self.mid_block = HunyuanVideo15MidBlock(in_channels=block_out_channels[-1])

        self.norm_out = HunyuanVideo15RMS_norm(block_out_channels[-1], images=False)
        self.conv_act = nn.SiLU()
        self.conv_out = HunyuanVideo15CausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(down_block, hidden_states)

            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states)

            hidden_states = self.mid_block(hidden_states)

        batch_size, _, frame, height, width = hidden_states.shape
        short_cut = hidden_states.view(batch_size, -1, self.group_size, frame, height, width).mean(dim=2)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        hidden_states += short_cut

        return hidden_states


class HunyuanVideo15Decoder3D(nn.Module):
    r"""
    Causal decoder for 3D video-like data used for HunyuanImage-1.5 Refiner.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (1024, 1024, 512, 256, 128),
        layers_per_block: int = 2,
        spatial_compression_ratio: int = 16,
        temporal_compression_ratio: int = 4,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat = block_out_channels[0] // self.in_channels

        self.conv_in = HunyuanVideo15CausalConv3d(self.in_channels, block_out_channels[0], kernel_size=3)
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = HunyuanVideo15MidBlock(in_channels=block_out_channels[0])

        # up
        input_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            output_channel = block_out_channels[i]

            add_spatial_upsample = i < np.log2(spatial_compression_ratio)
            add_temporal_upsample = i < np.log2(temporal_compression_ratio)
            if add_spatial_upsample or add_temporal_upsample:
                upsample_out_channels = block_out_channels[i + 1] if upsample_match_channel else output_channel
                up_block = HunyuanVideo15UpBlock3D(
                    num_layers=self.layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=upsample_out_channels,
                    add_temporal_upsample=add_temporal_upsample,
                )
                input_channel = upsample_out_channels
            else:
                up_block = HunyuanVideo15UpBlock3D(
                    num_layers=self.layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=None,
                    add_temporal_upsample=False,
                )
                input_channel = output_channel

            self.up_blocks.append(up_block)

        # out
        self.norm_out = HunyuanVideo15RMS_norm(block_out_channels[-1], images=False)
        self.conv_act = nn.SiLU()
        self.conv_out = HunyuanVideo15CausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states) + hidden_states.repeat_interleave(repeats=self.repeat, dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)

            for up_block in self.up_blocks:
                hidden_states = self._gradient_checkpointing_func(up_block, hidden_states)
        else:
            hidden_states = self.mid_block(hidden_states)

            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states)

        # post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderKLHunyuanVideo15(nn.Module, ParallelTiledVAE):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos. Used for
    HunyuanVideo-1.5.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Hunyuan15VAEConfig,
    ) -> None:
        nn.Module.__init__(self)
        ParallelTiledVAE.__init__(self, config)

        if config.load_encoder:
            self.encoder = HunyuanVideo15Encoder3D(
                in_channels=config.in_channels,
                out_channels=config.latent_channels * 2,
                block_out_channels=config.block_out_channels,
                layers_per_block=config.layers_per_block,
                temporal_compression_ratio=config.temporal_compression_ratio,
                spatial_compression_ratio=config.spatial_compression_ratio,
                downsample_match_channel=config.downsample_match_channel,
            )

        if config.load_decoder:
            self.decoder = HunyuanVideo15Decoder3D(
                in_channels=config.latent_channels,
                out_channels=config.out_channels,
                block_out_channels=list(reversed(config.block_out_channels)),
                layers_per_block=config.layers_per_block,
                temporal_compression_ratio=config.temporal_compression_ratio,
                spatial_compression_ratio=config.spatial_compression_ratio,
                upsample_match_channel=config.upsample_match_channel,
            )

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 2000 # Fill in a random large number, as hy1.5 vae does not use temporal tiling

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        dec = self.decoder(z)
        return dec

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec