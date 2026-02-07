# SPDX-License-Identifier: Apache-2.0
"""
GameCraft VAE building blocks - ported from official Hunyuan-GameCraft-1.0/hymm_sp/vae/unet_causal_3d_blocks.py.

Matches the official structure exactly for weight loading.
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def prepare_causal_attention_mask(
    n_frame: int, n_hw: int, dtype, device, batch_size: Optional[int] = None
):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class CausalConv3d(nn.Module):
    """Causal 3D convolution - matches official structure (has .conv)."""

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode: str = "replicate",
        disable_causal: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.pad_mode = pad_mode
        if isinstance(kernel_size, int):
            k = kernel_size
        else:
            k = kernel_size[0]
        if disable_causal:
            padding = (k // 2, k // 2, k // 2, k // 2, k // 2, k // 2)
        else:
            padding = (k // 2, k // 2, k // 2, k // 2, k - 1, 0)
        self.time_causal_padding = padding
        self.conv = nn.Conv3d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class DownsampleCausal3D(nn.Module):
    """Causal 3D downsampling - matches official (has .conv)."""

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        padding: int = 1,
        stride: Union[int, Tuple[int, int, int]] = 2,
        kernel_size: int = 3,
        bias: bool = True,
        disable_causal: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.conv = CausalConv3d(
            channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            disable_causal=disable_causal,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return self.conv(hidden_states)


class UpsampleCausal3D(nn.Module):
    """Causal 3D upsampling - matches official (has .conv when use_conv=True)."""

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        upsample_factor: Tuple[int, int, int] = (2, 2, 2),
        disable_causal: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.upsample_factor = upsample_factor
        self.disable_causal = disable_causal
        self.conv = CausalConv3d(
            channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=1,
            disable_causal=disable_causal,
            bias=bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        B, C, T, H, W = hidden_states.shape
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        if not self.disable_causal and T > 1:
            first_h, other_h = hidden_states.split((1, T - 1), dim=2)
            other_h = F.interpolate(
                other_h, scale_factor=self.upsample_factor, mode="nearest"
            )
            first_h = F.interpolate(
                first_h.squeeze(2), scale_factor=self.upsample_factor[1:], mode="nearest"
            ).unsqueeze(2)
            hidden_states = torch.cat((first_h, other_h), dim=2)
        else:
            hidden_states = F.interpolate(
                hidden_states, scale_factor=self.upsample_factor, mode="nearest"
            )

        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
        return self.conv(hidden_states)


class GameCraftVAEAttention(nn.Module):
    """Attention block matching official diffusers Attention structure (group_norm, to_q, to_k, to_v, to_out)."""

    def __init__(
        self,
        in_channels: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-6,
        norm_num_groups: Optional[int] = 32,
        bias: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.group_norm = nn.GroupNorm(
            norm_num_groups or in_channels, in_channels, eps=eps
        )
        self.to_q = nn.Linear(in_channels, inner_dim, bias=bias)
        self.to_k = nn.Linear(in_channels, inner_dim, bias=bias)
        self.to_v = nn.Linear(in_channels, inner_dim, bias=bias)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, in_channels, bias=bias))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = self.group_norm(
            hidden_states.permute(0, 2, 1)
        ).permute(0, 2, 1)

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)

        scale = self.dim_head**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1)
        hidden_states = torch.matmul(attn, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, -1)
        hidden_states = self.to_out(hidden_states) + residual
        return hidden_states


class ResnetBlockCausal3D(nn.Module):
    """ResNet block - matches official structure (conv1.conv, conv2.conv, norm1, norm2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: Optional[int] = None,
        eps: float = 1e-6,
        groups: int = 32,
        dropout: float = 0.0,
        non_linearity: str = "swish",
        disable_causal: bool = False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, 1, disable_causal=disable_causal)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, 1, disable_causal=disable_causal)
        self.dropout = nn.Dropout(dropout)
        self.conv_shortcut = (
            CausalConv3d(in_channels, out_channels, 1, 1, disable_causal=disable_causal)
            if in_channels != out_channels
            else None
        )
        self.nonlinearity = getattr(F, non_linearity, F.silu)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return (x + h) / 1.0


class UNetMidBlockCausal3D(nn.Module):
    """Mid block with resnets and optional attention - matches official structure."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: Optional[int] = None,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        disable_causal: bool = False,
        causal_attention: bool = False,
    ):
        super().__init__()
        self.add_attention = add_attention
        self.causal_attention = causal_attention

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        self.resnets.append(
            ResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=0.0,
                non_linearity=resnet_act_fn,
                disable_causal=disable_causal,
            )
        )

        for _ in range(num_layers):
            if add_attention:
                self.attentions.append(
                    GameCraftVAEAttention(
                        in_channels=in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        bias=True,
                    )
                )
            else:
                self.attentions.append(None)
            self.resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    non_linearity=resnet_act_fn,
                    disable_causal=disable_causal,
                )
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                B, C, T, H, W = hidden_states.shape
                hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")
                if self.causal_attention:
                    mask = prepare_causal_attention_mask(
                        T, H * W, hidden_states.dtype, hidden_states.device, batch_size=B
                    )
                else:
                    mask = None
                hidden_states = attn(hidden_states, attention_mask=mask)
                hidden_states = rearrange(
                    hidden_states, "b (f h w) c -> b c f h w", f=T, h=H, w=W
                )
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class DownEncoderBlockCausal3D(nn.Module):
    """Encoder down block - matches official structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_stride: Union[int, Tuple[int, int, int]] = 2,
        downsample_padding: int = 0,
        disable_causal: bool = False,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            inc = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlockCausal3D(
                    in_channels=inc,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    non_linearity=resnet_act_fn,
                    disable_causal=disable_causal,
                )
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                DownsampleCausal3D(
                    out_channels,
                    out_channels=out_channels,
                    padding=downsample_padding,
                    stride=downsample_stride,
                    disable_causal=disable_causal,
                )
            ])

    def forward(
        self, hidden_states: torch.Tensor, scale: float = 1.0
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, scale=scale)
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                hidden_states = ds(hidden_states, scale)
        return hidden_states


class UpDecoderBlockCausal3D(nn.Module):
    """Decoder up block - matches official structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_upsample: bool = True,
        upsample_scale_factor: Tuple[int, int, int] = (2, 2, 2),
        temb_channels: Optional[int] = None,
        disable_causal: bool = False,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            inc = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlockCausal3D(
                    in_channels=inc,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    non_linearity=resnet_act_fn,
                    disable_causal=disable_causal,
                )
            )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                UpsampleCausal3D(
                    out_channels,
                    out_channels=out_channels,
                    upsample_factor=upsample_scale_factor,
                    disable_causal=disable_causal,
                )
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb, scale=scale)
        if self.upsamplers is not None:
            for us in self.upsamplers:
                hidden_states = us(hidden_states)
        return hidden_states
