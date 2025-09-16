# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
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

import torch
import torch.nn as nn

from fastvideo.configs.models.vaes import LTXVAEConfig


from fastvideo.layers.layernorm import RMSNorm, FP32LayerNorm, LayerNormScaleShift
from fastvideo.layers.activation import get_act_fn
from fastvideo.layers.layernorm import ScaleResidual
from fastvideo.layers.activation import QuickGELU
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding, UnquantizedEmbeddingMethod

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.loaders import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin

from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from fastvideo.models.vaes.common import ParallelTiledVAE

from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution


class FastVideoLTXCausalConv3d(nn.Module):
    """FastVideo 3D causal convolution for LTX VAE."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        is_causal: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        # TODO: optimize convolution?
        # TODO: Add  parallelism across output channels
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]

        if self.is_causal:
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, time_kernel_size - 1, 1, 1))
            hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)
        else:
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            hidden_states = torch.concatenate([pad_left, hidden_states, pad_right], dim=2)

        hidden_states = self.conv(hidden_states)
        return hidden_states


class FastVideoLTXResnetBlock3d(nn.Module):
    r"""
    FastVideo 3D ResNet block for LTX VAE.
    
    TODO: fix optimizations
    - QuickGELU/FastSiLU activation
    - TOOD: add more parallelism
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        non_linearity: str = "swish",
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        # Use FastVideo activation functions
        if non_linearity == "swish":
            self.nonlinearity = get_act_fn("silu")
        elif non_linearity == "gelu":
            self.nonlinearity = get_act_fn("quick_gelu")
        else:
            
            self.nonlinearity = get_act_fn(non_linearity)

        # Use FastVideo RMSNorm
        self.norm1 = RMSNorm(in_channels, eps=1e-8, has_weight=elementwise_affine)

        self.conv1 = FastVideoLTXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, is_causal=is_causal
        )

        self.norm2 = RMSNorm(out_channels, eps=1e-8, has_weight=elementwise_affine)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = FastVideoLTXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, is_causal=is_causal
        )

        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            # Use FP32LayerNorm for better numerical stability in shortcut path
            self.norm3 = FP32LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            self.conv_shortcut = FastVideoLTXCausalConv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, is_causal=is_causal
            )

        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

        # # Use ScaleResidual for  residual connection
        # self.residual = ScaleResidual()

    def forward(
        self, inputs: torch.Tensor, temb: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.scale_shift_table is not None:
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]

        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class FastVideoLTXDownsampler3d(nn.Module):
    """FastVideo 3D downsampler for LTX VAE."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        is_causal: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.group_size = (in_channels * stride[0] * stride[1] * stride[2]) // out_channels

        out_channels = out_channels // (self.stride[0] * self.stride[1] * self.stride[2])

        # TODO: Add  parallelism ?
        self.conv = FastVideoLTXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.cat([hidden_states[:, :, : self.stride[0] - 1], hidden_states], dim=2)

        residual = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        residual = residual.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        residual = residual.unflatten(1, (-1, self.group_size))
        residual = residual.mean(dim=2)

        hidden_states = self.conv(hidden_states)
        hidden_states = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        hidden_states = hidden_states + residual

        return hidden_states


class FastVideoLTXUpsampler3d(nn.Module):
    """FastVideo 3D upsampler for LTX VAE."""
    
    def __init__(
        self,
        in_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        is_causal: bool = True,
        residual: bool = False,
        upscale_factor: int = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.residual = residual
        self.upscale_factor = upscale_factor

        out_channels = (in_channels * stride[0] * stride[1] * stride[2]) // upscale_factor

        # TODO: Add  parallelism ?
        self.conv = FastVideoLTXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        if self.residual:
            residual = hidden_states.reshape(
                batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
            )
            residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
            repeats = (self.stride[0] * self.stride[1] * self.stride[2]) // self.upscale_factor
            residual = residual.repeat(1, repeats, 1, 1, 1)
            residual = residual[:, :, self.stride[0] - 1 :]

        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        hidden_states = hidden_states[:, :, self.stride[0] - 1 :]

        if self.residual:
            hidden_states = hidden_states + residual

        return hidden_states


class FastVideoLTXDownBlock3D(nn.Module):
    r"""
    FastVideo down block for LTX VAE.
    
    TODO: add more optimizations:

    - add parallelism 
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        is_causal: bool = True,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                FastVideoLTXResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList(
                [
                    FastVideoLTXCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        is_causal=is_causal,
                    )
                ]
            )

        self.conv_out = None
        if in_channels != out_channels:
            self.conv_out = FastVideoLTXResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                is_causal=is_causal,
            )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        if self.conv_out is not None:
            hidden_states = self.conv_out(hidden_states, temb, generator)

        return hidden_states


class FastVideoLTX095DownBlock3D(nn.Module):
    r"""
    FastVideo down block for LTX VAE.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        is_causal: bool = True,
        downsample_type: str = "conv",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                FastVideoLTXResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList()

            if downsample_type == "conv":
                self.downsamplers.append(
                    FastVideoLTXCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        is_causal=is_causal,
                    )
                )
            elif downsample_type == "spatial":
                self.downsamplers.append(
                    FastVideoLTXDownsampler3d(
                        in_channels=in_channels, out_channels=out_channels, stride=(1, 2, 2), is_causal=is_causal
                    )
                )
            elif downsample_type == "temporal":
                self.downsamplers.append(
                    FastVideoLTXDownsampler3d(
                        in_channels=in_channels, out_channels=out_channels, stride=(2, 1, 1), is_causal=is_causal
                    )
                )
            elif downsample_type == "spatiotemporal":
                self.downsamplers.append(
                    FastVideoLTXDownsampler3d(
                        in_channels=in_channels, out_channels=out_channels, stride=(2, 2, 2), is_causal=is_causal
                    )
                )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class FastVideoLTXMidBlock3d(nn.Module):
    r"""
    FastVideo middle block for LTX VAE.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
    ) -> None:
        super().__init__()

        self.time_embedder = None
        if timestep_conditioning:
            # TODO: VocabParallelEmbedding?
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                FastVideoLTXResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        return hidden_states


class FastVideoLTXUpBlock3d(nn.Module):
    r"""
    FastVideo up block for LTX VAE.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        upsample_residual: bool = False,
        upscale_factor: int = 1,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.time_embedder = None
        if timestep_conditioning:
            # TODO: VocabParallelEmbedding?
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)

        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = FastVideoLTXResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                is_causal=is_causal,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
            )

        self.upsamplers = None
        if spatio_temporal_scale:
            self.upsamplers = nn.ModuleList(
                [
                    FastVideoLTXUpsampler3d(
                        out_channels * upscale_factor,
                        stride=(2, 2, 2),
                        is_causal=is_causal,
                        residual=upsample_residual,
                        upscale_factor=upscale_factor,
                    )
                ]
            )

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                FastVideoLTXResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states, temb, generator)

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        return hidden_states


class FastVideoLTXEncoder3d(nn.Module):
    r"""
    FastVideo encoder for LTX VAE.
    
    TODO: describe optimization:
    - add parallelism
    - replace more fastvideo normalization and activations?
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        down_block_types: Tuple[str, ...] = (
            "FastVideoLTXDownBlock3D",
            "FastVideoLTXDownBlock3D",
            "FastVideoLTXDownBlock3D",
            "FastVideoLTXDownBlock3D",
        ),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        downsample_type: Tuple[str, ...] = ("conv", "conv", "conv", "conv"),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels * patch_size**2

        output_channel = block_out_channels[0]

        # TODO: Add input dimension parallelism ?
        self.conv_in = FastVideoLTXCausalConv3d(
            in_channels=self.in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
        )

        # down blocks
        is_ltx_095 = down_block_types[-1] == "FastVideoLTX095DownBlock3D"
        num_block_out_channels = len(block_out_channels) - (1 if is_ltx_095 else 0)
        self.down_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            if not is_ltx_095:
                output_channel = block_out_channels[i + 1] if i + 1 < num_block_out_channels else block_out_channels[i]
            else:
                output_channel = block_out_channels[i + 1]

            if down_block_types[i] == "FastVideoLTXDownBlock3D":
                down_block = FastVideoLTXDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    resnet_eps=resnet_norm_eps,
                    spatio_temporal_scale=spatio_temporal_scaling[i],
                    is_causal=is_causal,
                )
            elif down_block_types[i] == "FastVideoLTX095DownBlock3D":
                down_block = FastVideoLTX095DownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    resnet_eps=resnet_norm_eps,
                    spatio_temporal_scale=spatio_temporal_scaling[i],
                    is_causal=is_causal,
                    downsample_type=downsample_type[i],
                )
            else:
                raise ValueError(f"Unknown down block type: {down_block_types[i]}")

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = FastVideoLTXMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[-1],
            resnet_eps=resnet_norm_eps,
            is_causal=is_causal,
        )

        # out
        self.norm_out = RMSNorm(output_channel, eps=1e-8, has_weight=False)


        self.conv_act = nn.SiLU()
        # TODO: Add output dimension parallelism 
        self.conv_out = FastVideoLTXCausalConv3d(
            in_channels=output_channel, out_channels=out_channels + 1, kernel_size=3, stride=1, is_causal=is_causal
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = hidden_states.reshape(
            batch_size, num_channels, post_patch_num_frames, p_t, post_patch_height, p, post_patch_width, p
        )
        # Thanks for driving me insane with the weird patching order :(
        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(down_block, hidden_states)

            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states)

            hidden_states = self.mid_block(hidden_states)

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        last_channel = hidden_states[:, -1:]
        last_channel = last_channel.repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        hidden_states = torch.cat([hidden_states, last_channel], dim=1)

        return hidden_states


class FastVideoLTXDecoder3d(nn.Module):
    r"""
    FastVideo decoder for LTX VAE.
    
    TODO: update optimizations:
    - add parallelism
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = False,
        inject_noise: Tuple[bool, ...] = (False, False, False, False),
        timestep_conditioning: bool = False,
        upsample_residual: Tuple[bool, ...] = (False, False, False, False),
        upsample_factor: Tuple[bool, ...] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels * patch_size**2

        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        inject_noise = tuple(reversed(inject_noise))
        upsample_residual = tuple(reversed(upsample_residual))
        upsample_factor = tuple(reversed(upsample_factor))
        output_channel = block_out_channels[0]

        # TODO: Add input dimension parallelism 
        self.conv_in = FastVideoLTXCausalConv3d(
            in_channels=in_channels, out_channels=output_channel, kernel_size=3, stride=1, is_causal=is_causal
        )

        self.mid_block = FastVideoLTXMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            is_causal=is_causal,
            inject_noise=inject_noise[0],
            timestep_conditioning=timestep_conditioning,
        )

        # up blocks
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]

            up_block = FastVideoLTXUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                is_causal=is_causal,
                inject_noise=inject_noise[i + 1],
                timestep_conditioning=timestep_conditioning,
                upsample_residual=upsample_residual[i],
                upscale_factor=upsample_factor[i],
            )

            self.up_blocks.append(up_block)

        # out
        self.norm_out = RMSNorm(output_channel, eps=1e-8, has_weight=False)
        self.conv_act = nn.SiLU()
        # TODO: Add output dimension parallelism 
        self.conv_out = FastVideoLTXCausalConv3d(
            in_channels=output_channel, out_channels=self.out_channels, kernel_size=3, stride=1, is_causal=is_causal
        )

        # timestep embedding
        self.time_embedder = None
        self.scale_shift_table = None
        self.timestep_scale_multiplier = None
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0, dtype=torch.float32))
            # TODO: VocabParallelEmbedding ?
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(output_channel * 2, 0)
            self.scale_shift_table = nn.Parameter(torch.randn(2, output_channel) / output_channel**0.5)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        if self.timestep_scale_multiplier is not None:
            temb = temb * self.timestep_scale_multiplier

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states, temb)

            for up_block in self.up_blocks:
                hidden_states = self._gradient_checkpointing_func(up_block, hidden_states, temb)
        else:
            hidden_states = self.mid_block(hidden_states, temb)

            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states, temb)

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
            temb = temb + self.scale_shift_table[None, ..., None, None, None]
            shift, scale = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, -1, p_t, p, p, num_frames, height, width)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return hidden_states

class AutoencoderKLLTXVideo(nn.Module):
    r"""
    FastVideo VAE model with KL loss for encoding images into latents and decoding latent representations into images.
        
    TODO: add more optimizations
    -  activation functions
    - add parallelism
    """

    _supports_gradient_checkpointing = True

    
    def __init__(
        self,
        config: LTXVAEConfig,
    ) -> None:
        nn.Module.__init__(self)
        ParallelTiledVAE.__init__(self, config)
        
        # Extract parameters from config
        # First check if config has arch_config (FastVideo style) or direct attributes (Diffusers style)
        if hasattr(config, 'arch_config'):
            # FastVideo style config
            arch_config = config.arch_config
            in_channels = getattr(arch_config, 'in_channels', 3)
            out_channels = getattr(arch_config, 'out_channels', 3)
            latent_channels = getattr(arch_config, 'latent_channels', 128)
            block_out_channels = getattr(arch_config, 'block_out_channels', (128, 256, 512, 512))
            layers_per_block = getattr(arch_config, 'layers_per_block', (4, 3, 3, 3, 4))
            spatio_temporal_scaling = getattr(arch_config, 'spatio_temporal_scaling', (True, True, True, False))
            patch_size = getattr(arch_config, 'patch_size', 4)
            patch_size_t = getattr(arch_config, 'patch_size_t', 1)
            resnet_norm_eps = getattr(arch_config, 'resnet_norm_eps', 1e-6)
            scaling_factor = getattr(arch_config, 'scaling_factor', 1.0)
            encoder_causal = getattr(arch_config, 'encoder_causal', True)
            decoder_causal = getattr(arch_config, 'decoder_causal', False)
            spatial_compression_ratio = getattr(arch_config, 'spatial_compression_ratio', None)
            temporal_compression_ratio = getattr(arch_config, 'temporal_compression_ratio', None)
            
            # Decoder specific parameters
            decoder_block_out_channels = getattr(arch_config, 'decoder_block_out_channels', block_out_channels)
            decoder_layers_per_block = getattr(arch_config, 'decoder_layers_per_block', layers_per_block)
            decoder_spatio_temporal_scaling = getattr(arch_config, 'decoder_spatio_temporal_scaling', spatio_temporal_scaling)
            decoder_inject_noise = getattr(arch_config, 'decoder_inject_noise', (False, False, False, False, False))
            downsample_type = getattr(arch_config, 'downsample_type', ("conv", "conv", "conv", "conv"))
            upsample_residual = getattr(arch_config, 'upsample_residual', (False, False, False, False))
            upsample_factor = getattr(arch_config, 'upsample_factor', (1, 1, 1, 1))
            timestep_conditioning = getattr(arch_config, 'timestep_conditioning', False)
        else:
            # Diffusers style config - attributes directly on config
            in_channels = getattr(config, 'in_channels', 3)
            out_channels = getattr(config, 'out_channels', 3)
            latent_channels = getattr(config, 'latent_channels', 128)
            block_out_channels = getattr(config, 'block_out_channels', (128, 256, 512, 512))
            layers_per_block = getattr(config, 'layers_per_block', (4, 3, 3, 3, 4))
            spatio_temporal_scaling = getattr(config, 'spatio_temporal_scaling', (True, True, True, False))
            patch_size = getattr(config, 'patch_size', 4)
            patch_size_t = getattr(config, 'patch_size_t', 1)
            resnet_norm_eps = getattr(config, 'resnet_norm_eps', 1e-6)
            scaling_factor = getattr(config, 'scaling_factor', 1.0)
            encoder_causal = getattr(config, 'encoder_causal', True)
            decoder_causal = getattr(config, 'decoder_causal', False)
            spatial_compression_ratio = getattr(config, 'spatial_compression_ratio', None)
            temporal_compression_ratio = getattr(config, 'temporal_compression_ratio', None)
            
            # Decoder specific parameters
            decoder_block_out_channels = getattr(config, 'decoder_block_out_channels', block_out_channels)
            decoder_layers_per_block = getattr(config, 'decoder_layers_per_block', layers_per_block)
            decoder_spatio_temporal_scaling = getattr(config, 'decoder_spatio_temporal_scaling', spatio_temporal_scaling)
            decoder_inject_noise = getattr(config, 'decoder_inject_noise', (False, False, False, False, False))
            downsample_type = getattr(config, 'downsample_type', ("conv", "conv", "conv", "conv"))
            upsample_residual = getattr(config, 'upsample_residual', (False, False, False, False))
            upsample_factor = getattr(config, 'upsample_factor', (1, 1, 1, 1))
            timestep_conditioning = getattr(config, 'timestep_conditioning', False)
        
        # Handle down_block_types - use FastVideo versions
        down_block_types = getattr(config, 'down_block_types', None)
        if down_block_types is None or all('LTXVideoDownBlock3D' in t for t in down_block_types):
            down_block_types = (
                "FastVideoLTXDownBlock3D",
                "FastVideoLTXDownBlock3D", 
                "FastVideoLTXDownBlock3D",
                "FastVideoLTXDownBlock3D",
            )

        # Create encoder and decoder with extracted parameters
        self.encoder = FastVideoLTXEncoder3d(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            downsample_type=downsample_type,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=encoder_causal,
        )
        
        self.decoder = FastVideoLTXDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decoder_block_out_channels,
            spatio_temporal_scaling=decoder_spatio_temporal_scaling,
            layers_per_block=decoder_layers_per_block,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=decoder_causal,
            timestep_conditioning=timestep_conditioning,
            inject_noise=decoder_inject_noise,
            upsample_residual=upsample_residual,
            upsample_factor=upsample_factor,
        )

        # Register buffers for latent normalization
        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

        # Set compression ratios
        self.spatial_compression_ratio = (
            patch_size * 2 ** sum(spatio_temporal_scaling)
            if spatial_compression_ratio is None
            else spatial_compression_ratio
        )
        self.temporal_compression_ratio = (
            patch_size_t * 2 ** sum(spatio_temporal_scaling)
            if temporal_compression_ratio is None
            else temporal_compression_ratio
        )

        # Memory optimization settings
        self.use_slicing = False
        self.use_tiling = False
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

        # Batch sizes for memory management
        self.num_sample_frames_batch_size = 16
        self.num_latent_frames_batch_size = 2

        # Tiling parameters
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512
        self.tile_sample_min_num_frames = 16
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448
        self.tile_sample_stride_num_frames = 8
        
        # Store config for compatibility
        self.config = config

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_num_frames: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_sample_stride_num_frames: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_framewise_encoding and num_frames > self.tile_sample_min_num_frames:
            return self._temporal_tiled_encode(x)

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        enc = self.encoder(x)
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor] = None, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio

        if self.use_framewise_decoding and num_frames > tile_latent_min_num_frames:
            return self._temporal_tiled_decode(z, temb, return_dict=return_dict)

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, temb, return_dict=return_dict)

        dec = self.decoder(z, temb)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor] = None, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of images.
        """
        if self.use_slicing and z.shape[0] > 1:
            if temb is not None:
                decoded_slices = [
                    self._decode(z_slice, t_slice).sample for z_slice, t_slice in (z.split(1), temb.split(1))
                ]
            else:
                decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, temb).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder."""
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                time = self.encoder(
                    x[:, :, :, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                )

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor], return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.
        """

        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                time = self.decoder(z[:, :, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width], temb)

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def _temporal_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1

        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames

        row = []
        for i in range(0, num_frames, self.tile_sample_stride_num_frames):
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling and (height > self.tile_sample_min_height or width > self.tile_sample_min_width):
                tile = self.tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                result_row.append(tile[:, :, :tile_latent_stride_num_frames, :, :])
            else:
                result_row.append(tile[:, :, : tile_latent_stride_num_frames + 1, :, :])

        enc = torch.cat(result_row, dim=2)[:, :, :latent_num_frames]
        return enc

    def _temporal_tiled_decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor], return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames

        row = []
        for i in range(0, num_frames, tile_latent_stride_num_frames):
            tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (tile.shape[-1] > tile_latent_min_width or tile.shape[-2] > tile_latent_min_height):
                decoded = self.tiled_decode(tile, temb, return_dict=True).sample
            else:
                decoded = self.decoder(tile, temb)
            if i > 0:
                decoded = decoded[:, :, :-1, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                tile = tile[:, :, : self.tile_sample_stride_num_frames, :, :]
                result_row.append(tile)
            else:
                result_row.append(tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :])

        dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, temb)
        if not return_dict:
            return (dec.sample,)
        return dec

