# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Team. All rights reserved.
# Adapted from: huggingface/diffusers `Encoder`/`Decoder` VAE components
# at the installed 0.36.0 source surface used by Flux2 Klein.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2._vendor.models.vaes.common import DiagonalGaussianDistribution


@dataclass
class AutoencoderKLOutput:
    latent_dist: DiagonalGaussianDistribution

    def __getitem__(self, idx: int):
        return (self.latent_dist,)[idx]

    def __getattr__(self, name: str):
        # Existing local Flux2 parity tests used `vae.encode(x).mean` while
        # diffusers-style callers use `vae.encode(x).latent_dist.mean`.
        return getattr(self.latent_dist, name)


@dataclass
class DecoderOutput:
    sample: torch.Tensor
    commit_loss: Optional[torch.Tensor] = None

    def __getitem__(self, idx: int):
        return (self.sample, self.commit_loss)[idx]


def get_activation(act_fn: str) -> nn.Module:
    if act_fn in ("swish", "silu"):
        return nn.SiLU()
    if act_fn == "mish":
        return nn.Mish()
    if act_fn == "gelu":
        return nn.GELU()
    if act_fn == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation function: {act_fn}")


class AttnProcessor:
    def __call__(self, attn: "Attention", hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        return attn._forward(hidden_states, temb=temb)


class AttnAddedKVProcessor(AttnProcessor):
    pass


ADDED_KV_ATTENTION_PROCESSORS = frozenset({AttnAddedKVProcessor})
CROSS_ATTENTION_PROCESSORS = frozenset({AttnProcessor})


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_softmax: bool = False,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        **_: object,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self._from_deprecated_attn_block = _from_deprecated_attn_block
        self.spatial_norm = None
        if spatial_norm_dim is not None:
            raise ValueError("Flux2 VAE does not use spatial attention norm in this port")
        self.group_norm = (
            nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
            if norm_num_groups is not None
            else None
        )
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=out_bias), nn.Dropout(dropout)])
        self.processor = AttnProcessor()

    def set_processor(self, processor: AttnProcessor) -> None:
        self.processor = processor

    def get_processor(self) -> AttnProcessor:
        return self.processor

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.processor(self, hidden_states, temb=temb)

    def _forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        batch_size, channel, height, width = hidden_states.shape
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        if self.upcast_softmax:
            query = query.float()
            key = key.float()
            value = value.float()
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, scale=self.scale)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, height * width, self.inner_dim)
        hidden_states = hidden_states.to(self.to_out[0].weight.dtype)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, channel, height, width)
        if self.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / self.rescale_output_factor


class Downsample2D(nn.Module):
    def __init__(self, channels: int, use_conv: bool = False, out_channels: Optional[int] = None, padding: int = 1, name: str = "conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.name = name
        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=2, stride=2)
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(self, channels: int, use_conv: bool = False, out_channels: Optional[int] = None, name: str = "conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = False
        self.name = name
        self.interpolate = True
        conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1) if use_conv else None
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.float()
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
        if self.use_conv:
            hidden_states = self.conv(hidden_states) if self.name == "conv" else self.Conv2d_0(hidden_states)
        return hidden_states


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: Optional[int] = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        **_: object,
    ):
        super().__init__()
        if time_embedding_norm not in ("default", "scale_shift"):
            raise ValueError(f"unknown time_embedding_norm: {time_embedding_norm}")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        if groups_out is None:
            groups_out = groups
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels if time_embedding_norm == "default" else 2 * out_channels)
        else:
            self.time_emb_proj = None
        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = get_activation(non_linearity)
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias)

    def forward(self, input_tensor: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        hidden_states = self.norm1(input_tensor)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if self.time_emb_proj is not None and temb is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        else:
            if temb is None:
                raise ValueError("temb cannot be None for scale_shift")
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous() if self.training else input_tensor)
        return (input_tensor + hidden_states) / self.output_scale_factor


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: Optional[int],
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        if resnet_time_scale_shift == "spatial":
            raise ValueError("Flux2 VAE does not use spatial resnet conditioning in this port")
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention
        if attn_groups is None:
            attn_groups = resnet_groups
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        if attention_head_dim is None:
            attention_head_dim = in_channels
        for _ in range(num_layers):
            attentions.append(
                Attention(
                    in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=attn_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
                if self.add_attention
                else None
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, num_layers: int = 1, resnet_eps: float = 1e-6, resnet_act_fn: str = "swish", resnet_groups: int = 32, add_downsample: bool = True, downsample_padding: int = 1, **_: object):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=None,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
            )
            for i in range(num_layers)
        ])
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op")]) if add_downsample else None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class AttnDownEncoderBlock2D(DownEncoderBlock2D):
    def __init__(self, in_channels: int, out_channels: int, attention_head_dim: int = 1, **kwargs: object):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        resnet_groups = int(kwargs.get("resnet_groups", 32))
        resnet_eps = float(kwargs.get("resnet_eps", 1e-6))
        output_scale_factor = float(kwargs.get("output_scale_factor", 1.0))
        if attention_head_dim is None:
            attention_head_dim = out_channels
        self.attentions = nn.ModuleList([
            Attention(out_channels, heads=out_channels // attention_head_dim, dim_head=attention_head_dim, rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=resnet_groups, residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True)
            for _ in self.resnets
        ])

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = attn(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, num_layers: int = 1, resnet_eps: float = 1e-6, resnet_act_fn: str = "swish", resnet_groups: int = 32, add_upsample: bool = True, temb_channels: Optional[int] = None, **_: object):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
            )
            for i in range(num_layers)
        ])
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]) if add_upsample else None
        self.resolution_idx = None

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class AttnUpDecoderBlock2D(UpDecoderBlock2D):
    def __init__(self, in_channels: int, out_channels: int, attention_head_dim: int = 1, **kwargs: object):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        resnet_groups = int(kwargs.get("resnet_groups", 32))
        resnet_eps = float(kwargs.get("resnet_eps", 1e-6))
        output_scale_factor = float(kwargs.get("output_scale_factor", 1.0))
        if attention_head_dim is None:
            attention_head_dim = out_channels
        self.attentions = nn.ModuleList([
            Attention(out_channels, heads=out_channels // attention_head_dim, dim_head=attention_head_dim, rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=resnet_groups, residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True)
            for _ in self.resnets
        ])

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb=temb)
            hidden_states = attn(hidden_states, temb=temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


def get_down_block(down_block_type: str, **kwargs: object) -> nn.Module:
    if down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(**kwargs)
    if down_block_type == "AttnDownEncoderBlock2D":
        return AttnDownEncoderBlock2D(**kwargs)
    raise ValueError(f"Unsupported Flux2 VAE down block type: {down_block_type}")


def get_up_block(up_block_type: str, **kwargs: object) -> nn.Module:
    kwargs.pop("prev_output_channel", None)
    kwargs.pop("resolution_idx", None)
    if up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(**kwargs)
    if up_block_type == "AttnUpDecoderBlock2D":
        return AttnUpDecoderBlock2D(**kwargs)
    raise ValueError(f"Unsupported Flux2 VAE up block type: {up_block_type}")


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",), block_out_channels: Tuple[int, ...] = (64,), layers_per_block: int = 2, norm_num_groups: int = 32, act_fn: str = "silu", double_z: bool = True, mid_block_add_attention: bool = True):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            self.down_blocks.append(get_down_block(down_block_type, num_layers=self.layers_per_block, in_channels=input_channel, out_channels=output_channel, add_downsample=not is_final_block, resnet_eps=1e-6, downsample_padding=0, resnet_act_fn=act_fn, resnet_groups=norm_num_groups, attention_head_dim=output_channel, temb_channels=None))
        self.mid_block = UNetMidBlock2D(in_channels=block_out_channels[-1], resnet_eps=1e-6, resnet_act_fn=act_fn, output_scale_factor=1, resnet_time_scale_shift="default", attention_head_dim=block_out_channels[-1], resnet_groups=norm_num_groups, temb_channels=None, add_attention=mid_block_add_attention)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        return self.conv_out(sample)


class Decoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",), block_out_channels: Tuple[int, ...] = (64,), layers_per_block: int = 2, norm_num_groups: int = 32, act_fn: str = "silu", norm_type: str = "group", mid_block_add_attention: bool = True):
        super().__init__()
        if norm_type != "group":
            raise ValueError("Flux2 VAE Decoder only supports group norm in this port")
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.up_blocks = nn.ModuleList([])
        self.mid_block = UNetMidBlock2D(in_channels=block_out_channels[-1], resnet_eps=1e-6, resnet_act_fn=act_fn, output_scale_factor=1, resnet_time_scale_shift="default", attention_head_dim=block_out_channels[-1], resnet_groups=norm_num_groups, temb_channels=None, add_attention=mid_block_add_attention)
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            self.up_blocks.append(get_up_block(up_block_type, num_layers=self.layers_per_block + 1, in_channels=prev_output_channel, out_channels=output_channel, prev_output_channel=prev_output_channel, add_upsample=not is_final_block, resnet_eps=1e-6, resnet_act_fn=act_fn, resnet_groups=norm_num_groups, attention_head_dim=output_channel, temb_channels=None, resnet_time_scale_shift=norm_type))
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, latent_embeds)
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        return self.conv_out(sample)
