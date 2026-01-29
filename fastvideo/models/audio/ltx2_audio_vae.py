# SPDX-License-Identifier: Apache-2.0
"""
Native LTX-2 Audio VAE and Vocoder implementation for FastVideo.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, List, NamedTuple, Set, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Enums
# =============================================================================


class NormType(Enum):
    """Normalization layer types: GROUP (GroupNorm) or PIXEL (per-location RMS norm)."""
    GROUP = "group"
    PIXEL = "pixel"


class AttentionType(Enum):
    """Enum for specifying the attention mechanism type."""
    VANILLA = "vanilla"
    LINEAR = "linear"
    NONE = "none"


class CausalityAxis(Enum):
    """Enum for specifying the causality axis in causal convolutions."""
    NONE = None
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width-compatibility"


# =============================================================================
# Audio Latent Shape
# =============================================================================


class AudioLatentShape(NamedTuple):
    """Shape of audio in VAE latent space: (batch, channels, frames, mel_bins)."""
    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> AudioLatentShape:
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )


# =============================================================================
# Constants
# =============================================================================


LATENT_DOWNSAMPLE_FACTOR = 4
LRELU_SLOPE = 0.1


# =============================================================================
# Normalization Layers
# =============================================================================


class PixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x ** 2, dim=self.dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


def build_normalization_layer(
    in_channels: int,
    *,
    num_groups: int = 32,
    normtype: NormType = NormType.GROUP,
) -> nn.Module:
    """Create a normalization layer based on the normalization type."""
    if normtype == NormType.GROUP:
        return nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
    if normtype == NormType.PIXEL:
        return PixelNorm(dim=1, eps=1e-6)
    raise ValueError(f"Invalid normalization type: {normtype}")


# =============================================================================
# Per-Channel Statistics
# =============================================================================


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").to(x)) + self.get_buffer(
            "mean-of-means"
        ).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer(
            "std-of-means"
        ).to(x)


# =============================================================================
# Audio Patchifier
# =============================================================================


class AudioPatchifier:
    """Simple patchifier for audio latents."""

    def __init__(
        self,
        patch_size: int = 1,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
    ):
        self.patch_size = patch_size
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal

    def patchify(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """Flatten audio latent tensor along time: (B, C, T, F) -> (B, T, C*F)."""
        return einops.rearrange(audio_latents, "b c t f -> b t (c f)")

    def unpatchify(
        self, audio_latents: torch.Tensor, output_shape: AudioLatentShape
    ) -> torch.Tensor:
        """Restore (B, C, T, F) from flattened patches: (B, T, C*F) -> (B, C, T, F)."""
        return einops.rearrange(
            audio_latents,
            "b t (c f) -> b c t f",
            c=output_shape.channels,
            f=output_shape.mel_bins,
        )


# =============================================================================
# Causal 2D Convolution
# =============================================================================


class CausalConv2d(nn.Module):
    """
    A causal 2D convolution.
    Ensures output at time t only depends on inputs at time t and earlier.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()

        self.causality_axis = causality_axis

        # Ensure kernel_size and dilation are tuples
        kernel_size = nn.modules.utils._pair(kernel_size)
        dilation = nn.modules.utils._pair(dilation)

        # Calculate padding dimensions
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        # The padding tuple for F.pad is (pad_left, pad_right, pad_top, pad_bottom)
        if self.causality_axis == CausalityAxis.NONE:
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis in (CausalityAxis.WIDTH, CausalityAxis.WIDTH_COMPATIBILITY):
            self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis == CausalityAxis.HEIGHT:
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")

        # The internal convolution layer uses no padding, as we handle it manually
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Tuple[int, int],
    stride: int = 1,
    padding: Tuple[int, int, int, int] | None = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis | None = None,
) -> nn.Module:
    """Create a 2D convolution layer that can be either causal or non-causal."""
    if causality_axis is not None:
        return CausalConv2d(
            in_channels, out_channels, kernel_size, stride, dilation, groups, bias, causality_axis
        )
    else:
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )


# =============================================================================
# Attention Blocks
# =============================================================================


class AttnBlock(nn.Module):
    """Vanilla self-attention block for 2D features."""

    def __init__(
        self,
        in_channels: int,
        norm_type: NormType = NormType.GROUP,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).contiguous()
        q = q.permute(0, 2, 1).contiguous()  # b, hw, c
        k = k.reshape(b, c, h * w).contiguous()  # b, c, hw
        w_ = torch.bmm(q, k).contiguous()  # b, hw, hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # Attend to values
        v = v.reshape(b, c, h * w).contiguous()
        w_ = w_.permute(0, 2, 1).contiguous()  # b, hw, hw
        h_ = torch.bmm(v, w_).contiguous()  # b, c, hw
        h_ = h_.reshape(b, c, h, w).contiguous()

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(
    in_channels: int,
    attn_type: AttentionType = AttentionType.VANILLA,
    norm_type: NormType = NormType.GROUP,
) -> nn.Module:
    """Factory function for attention blocks."""
    if attn_type == AttentionType.VANILLA:
        return AttnBlock(in_channels, norm_type=norm_type)
    elif attn_type == AttentionType.NONE:
        return nn.Identity()
    elif attn_type == AttentionType.LINEAR:
        raise NotImplementedError(f"Attention type {attn_type.value} is not supported yet.")
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")


# =============================================================================
# ResNet Blocks (2D for Audio VAE)
# =============================================================================


class ResnetBlock(nn.Module):
    """2D ResNet block for audio VAE."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and norm_type == NormType.GROUP:
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = build_normalization_layer(in_channels, normtype=norm_type)
        self.non_linearity = nn.SiLU()
        self.conv1 = make_conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
        )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = build_normalization_layer(out_channels, normtype=norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = make_conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
                )
            else:
                self.nin_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, causality_axis=causality_axis
                )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)

        return x + h


# =============================================================================
# Vocoder ResBlocks (1D)
# =============================================================================


class ResBlock1(nn.Module):
    """1D ResBlock for vocoder with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding="same"),
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding="same"),
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding="same"),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding="same"),
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding="same"),
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding="same"),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    """1D ResBlock for vocoder (simpler version)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int] = (1, 3),
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding="same"),
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding="same"),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x


# =============================================================================
# Downsampling
# =============================================================================


class Downsample(nn.Module):
    """Downsampling layer with strided convolution or average pooling."""

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and not self.with_conv:
            raise ValueError("causality is only supported when `with_conv=True`.")

        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # Padding tuple is in the order: (left, right, top, bottom)
            if self.causality_axis == CausalityAxis.NONE:
                pad = (0, 1, 0, 1)
            elif self.causality_axis == CausalityAxis.WIDTH:
                pad = (2, 0, 0, 1)
            elif self.causality_axis == CausalityAxis.HEIGHT:
                pad = (0, 1, 2, 0)
            elif self.causality_axis == CausalityAxis.WIDTH_COMPATIBILITY:
                pad = (1, 0, 0, 1)
            else:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)

        return x


def build_downsampling_path(
    *,
    ch: int,
    ch_mult: Tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: Set[int],
    resamp_with_conv: bool,
) -> Tuple[nn.ModuleList, int]:
    """Build the downsampling path with residual blocks, attention, and downsampling layers."""
    down_modules = nn.ModuleList()
    curr_res = resolution
    in_ch_mult = (1, *tuple(ch_mult))
    block_in = ch

    for i_level in range(num_resolutions):
        block = nn.ModuleList()
        attn = nn.ModuleList()
        block_in = ch * in_ch_mult[i_level]
        block_out = ch * ch_mult[i_level]

        for _ in range(num_res_blocks):
            block.append(
                ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    causality_axis=causality_axis,
                )
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                attn.append(make_attn(block_in, attn_type=attn_type, norm_type=norm_type))

        down = nn.Module()
        down.block = block
        down.attn = attn
        if i_level != num_resolutions - 1:
            down.downsample = Downsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res = curr_res // 2
        down_modules.append(down)

    return down_modules, block_in


# =============================================================================
# Upsampling
# =============================================================================


class Upsample(nn.Module):
    """Upsampling layer with nearest-neighbor interpolation and optional convolution."""

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, causality_axis=causality_axis
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            # Drop FIRST element in the causal axis to undo encoder's padding
            if self.causality_axis == CausalityAxis.NONE:
                pass
            elif self.causality_axis == CausalityAxis.HEIGHT:
                x = x[:, :, 1:, :]
            elif self.causality_axis == CausalityAxis.WIDTH:
                x = x[:, :, :, 1:]
            elif self.causality_axis == CausalityAxis.WIDTH_COMPATIBILITY:
                pass
            else:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        return x


def build_upsampling_path(
    *,
    ch: int,
    ch_mult: Tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: Set[int],
    resamp_with_conv: bool,
    initial_block_channels: int,
) -> Tuple[nn.ModuleList, int]:
    """Build the upsampling path with residual blocks, attention, and upsampling layers."""
    up_modules = nn.ModuleList()
    block_in = initial_block_channels
    curr_res = resolution // (2 ** (num_resolutions - 1))

    for level in reversed(range(num_resolutions)):
        stage = nn.Module()
        stage.block = nn.ModuleList()
        stage.attn = nn.ModuleList()
        block_out = ch * ch_mult[level]

        for _ in range(num_res_blocks + 1):
            stage.block.append(
                ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    causality_axis=causality_axis,
                )
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                stage.attn.append(make_attn(block_in, attn_type=attn_type, norm_type=norm_type))

        if level != 0:
            stage.upsample = Upsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res *= 2

        up_modules.insert(0, stage)

    return up_modules, block_in


# =============================================================================
# Mid Block
# =============================================================================


def build_mid_block(
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    add_attention: bool,
) -> nn.Module:
    """Build the middle block with two ResNet blocks and optional attention."""
    mid = nn.Module()
    mid.block_1 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    mid.attn_1 = (
        make_attn(channels, attn_type=attn_type, norm_type=norm_type)
        if add_attention
        else nn.Identity()
    )
    mid.block_2 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    return mid


def run_mid_block(mid: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run features through the middle block."""
    features = mid.block_1(features, temb=None)
    features = mid.attn_1(features)
    return mid.block_2(features, temb=None)


# =============================================================================
# Audio Encoder
# =============================================================================


class AudioEncoder(nn.Module):
    """
    Encoder that compresses audio spectrograms into latent representations.
    """

    def __init__(
        self,
        *,
        ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Set[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        attn_type: AttentionType = AttentionType.VANILLA,
        mid_block_add_attention: bool = True,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
        is_causal: bool = True,
        mel_bins: int = 64,
        **_ignore_kwargs,
    ) -> None:
        super().__init__()

        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft
        self.is_causal = is_causal
        self.mel_bins = mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.double_z = double_z
        self.norm_type = norm_type
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        # Input convolution
        self.conv_in = make_conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

        self.non_linearity = nn.SiLU()

        # Downsampling path
        self.down, block_in = build_downsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
        )

        # Mid block
        self.mid = build_mid_block(
            channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )

        # Output layers
        self.norm_out = build_normalization_layer(block_in, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Encode audio spectrogram into latent representations.
        Args:
            spectrogram: Input spectrogram of shape (batch, channels, time, frequency)
        Returns:
            Encoded latent representation of shape (batch, channels, frames, mel_bins)
        """
        h = self.conv_in(spectrogram)
        h = self._run_downsampling_path(h)
        h = run_mid_block(self.mid, h)
        h = self._finalize_output(h)

        return self._normalize_latents(h)

    def _run_downsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx in range(self.num_res_blocks):
                h = stage.block[block_idx](h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)

            if level != self.num_resolutions - 1:
                h = stage.downsample(h)

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm_out(h)
        h = self.non_linearity(h)
        return self.conv_out(h)

    def _normalize_latents(self, latent_output: torch.Tensor) -> torch.Tensor:
        """Normalize encoder latents using per-channel statistics."""
        means = torch.chunk(latent_output, 2, dim=1)[0]
        latent_shape = AudioLatentShape(
            batch=means.shape[0],
            channels=means.shape[1],
            frames=means.shape[2],
            mel_bins=means.shape[3],
        )
        latent_patched = self.patchifier.patchify(means)
        latent_normalized = self.per_channel_statistics.normalize(latent_patched)
        return self.patchifier.unpatchify(latent_normalized, latent_shape)


# =============================================================================
# Audio Decoder
# =============================================================================


class AudioDecoder(nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.
    """

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Set[int],
        resolution: int,
        z_channels: int,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
        dropout: float = 0.0,
        mid_block_add_attention: bool = True,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = None,
    ) -> None:
        super().__init__()

        # Internal behavioural defaults
        resamp_with_conv = True
        attn_type = AttentionType.VANILLA

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.out_ch = out_ch
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.z_channels = z_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        base_block_channels = ch * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, z_channels, base_resolution, base_resolution)

        self.conv_in = make_conv2d(
            z_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )
        self.non_linearity = nn.SiLU()

        self.mid = build_mid_block(
            channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )

        self.up, final_block_channels = build_upsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
            initial_block_channels=base_block_channels,
        )

        self.norm_out = build_normalization_layer(final_block_channels, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            final_block_channels, out_ch, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features back to audio spectrograms.
        Args:
            sample: Encoded latent representation of shape (batch, channels, frames, mel_bins)
        Returns:
            Reconstructed audio spectrogram of shape (batch, channels, time, frequency)
        """
        sample, target_shape = self._denormalize_latents(sample)

        h = self.conv_in(sample)
        h = run_mid_block(self.mid, h)
        h = self._run_upsampling_path(h)
        h = self._finalize_output(h)

        return self._adjust_output_shape(h, target_shape)

    def _denormalize_latents(
        self, sample: torch.Tensor
    ) -> Tuple[torch.Tensor, AudioLatentShape]:
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[1],
            frames=sample.shape[2],
            mel_bins=sample.shape[3],
        )

        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.per_channel_statistics.un_normalize(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis != CausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins,
        )

        return sample, target_shape

    def _adjust_output_shape(
        self,
        decoded_output: torch.Tensor,
        target_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """Adjust output shape to match target dimensions for variable-length audio."""
        _, _, current_time, current_freq = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        # Crop first
        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        # Calculate padding needed
        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]

        # Apply padding if needed
        if time_padding_needed > 0 or freq_padding_needed > 0:
            padding = (
                0,
                max(freq_padding_needed, 0),
                0,
                max(time_padding_needed, 0),
            )
            decoded_output = F.pad(decoded_output, padding)

        # Final safety crop
        decoded_output = decoded_output[:, :target_channels, :target_time, :target_freq]

        return decoded_output

    def _run_upsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                h = block(h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)

            if level != 0 and hasattr(stage, "upsample"):
                h = stage.upsample(h)

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.non_linearity(h)
        h = self.conv_out(h)
        return torch.tanh(h) if self.tanh_out else h


# =============================================================================
# Vocoder
# =============================================================================


class Vocoder(nn.Module):
    """
    Vocoder model for synthesizing audio from Mel spectrograms.
    """

    def __init__(
        self,
        resblock_kernel_sizes: List[int] | None = None,
        upsample_rates: List[int] | None = None,
        upsample_kernel_sizes: List[int] | None = None,
        resblock_dilation_sizes: List[List[int]] | None = None,
        upsample_initial_channel: int = 1024,
        stereo: bool = True,
        resblock: str = "1",
        output_sample_rate: int = 24000,
    ):
        super().__init__()

        # Initialize default values
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [6, 5, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sample_rate = output_sample_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        in_channels = 128 if stereo else 64
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        resblock_class = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (stride, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size,
                    stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i, _ in enumerate(self.ups):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock_class(ch, kernel_size, dilations))

        out_channels = 2 if stereo else 1
        final_channels = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = nn.Conv1d(final_channels, out_channels, 7, 1, padding=3)

        self.upsample_factor = math.prod(layer.stride[0] for layer in self.ups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vocoder.
        Args:
            x: Input Mel spectrogram tensor of shape (batch, channels, time, mel_bins)
        Returns:
            Audio waveform tensor of shape (batch, out_channels, audio_length)
        """
        x = x.transpose(2, 3)  # (batch, channels, time, mel_bins) -> (batch, channels, mel_bins, time)

        if x.dim() == 4:  # stereo
            assert x.shape[1] == 2, "Input must have 2 channels for stereo"
            x = einops.rearrange(x, "b s c t -> b (s c) t")

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels

            # Evaluate all resblocks and average
            block_outputs = torch.stack(
                [self.resblocks[idx](x) for idx in range(start, end)],
                dim=0,
            )

            x = block_outputs.mean(dim=0)

        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)


# =============================================================================
# Configurators
# =============================================================================


class AudioEncoderConfigurator:
    """Factory for AudioEncoder from checkpoint config."""

    @classmethod
    def from_config(cls, config: dict) -> AudioEncoder:
        audio_vae_cfg = config.get("audio_vae", {})
        model_cfg = audio_vae_cfg.get("model", {})
        model_params = model_cfg.get("params", {})
        ddconfig = model_params.get("ddconfig", {})
        preprocessing_cfg = audio_vae_cfg.get("preprocessing", {})
        stft_cfg = preprocessing_cfg.get("stft", {})
        mel_cfg = preprocessing_cfg.get("mel", {})
        variables_cfg = audio_vae_cfg.get("variables", {})

        sample_rate = model_params.get("sampling_rate", 16000)
        mel_hop_length = stft_cfg.get("hop_length", 160)
        n_fft = stft_cfg.get("filter_length", 1024)
        is_causal = stft_cfg.get("causal", True)
        mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels") or variables_cfg.get("mel_bins")

        return AudioEncoder(
            ch=ddconfig.get("ch", 128),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=ddconfig.get("num_res_blocks", 2),
            attn_resolutions=set(ddconfig.get("attn_resolutions", {8, 16, 32})),
            resolution=ddconfig.get("resolution", 256),
            z_channels=ddconfig.get("z_channels", 8),
            double_z=ddconfig.get("double_z", True),
            dropout=ddconfig.get("dropout", 0.0),
            resamp_with_conv=ddconfig.get("resamp_with_conv", True),
            in_channels=ddconfig.get("in_channels", 2),
            attn_type=AttentionType(ddconfig.get("attn_type", "vanilla")),
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", True),
            norm_type=NormType(ddconfig.get("norm_type", "pixel")),
            causality_axis=CausalityAxis(ddconfig.get("causality_axis", "height")),
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            n_fft=n_fft,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )


class AudioDecoderConfigurator:
    """Factory for AudioDecoder from checkpoint config."""

    @classmethod
    def from_config(cls, config: dict) -> AudioDecoder:
        audio_vae_cfg = config.get("audio_vae", {})
        model_cfg = audio_vae_cfg.get("model", {})
        model_params = model_cfg.get("params", {})
        ddconfig = model_params.get("ddconfig", {})
        preprocessing_cfg = audio_vae_cfg.get("preprocessing", {})
        stft_cfg = preprocessing_cfg.get("stft", {})
        mel_cfg = preprocessing_cfg.get("mel", {})
        variables_cfg = audio_vae_cfg.get("variables", {})

        sample_rate = model_params.get("sampling_rate", 16000)
        mel_hop_length = stft_cfg.get("hop_length", 160)
        is_causal = stft_cfg.get("causal", True)
        mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels") or variables_cfg.get("mel_bins")

        return AudioDecoder(
            ch=ddconfig.get("ch", 128),
            out_ch=ddconfig.get("out_ch", 2),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=ddconfig.get("num_res_blocks", 2),
            attn_resolutions=set(ddconfig.get("attn_resolutions", {8, 16, 32})),
            resolution=ddconfig.get("resolution", 256),
            z_channels=ddconfig.get("z_channels", 8),
            norm_type=NormType(ddconfig.get("norm_type", "pixel")),
            causality_axis=CausalityAxis(ddconfig.get("causality_axis", "height")),
            dropout=ddconfig.get("dropout", 0.0),
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", True),
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )


class VocoderConfigurator:
    """Factory for Vocoder from checkpoint config."""

    @classmethod
    def from_config(cls, config: dict) -> Vocoder:
        vocoder_cfg = config.get("vocoder", {})
        return Vocoder(
            resblock_kernel_sizes=vocoder_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
            upsample_rates=vocoder_cfg.get("upsample_rates", [6, 5, 2, 2, 2]),
            upsample_kernel_sizes=vocoder_cfg.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
            resblock_dilation_sizes=vocoder_cfg.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            upsample_initial_channel=vocoder_cfg.get("upsample_initial_channel", 1024),
            stereo=vocoder_cfg.get("stereo", True),
            resblock=vocoder_cfg.get("resblock", "1"),
            output_sample_rate=vocoder_cfg.get("output_sample_rate", 24000),
        )


# =============================================================================
# Public API Wrappers
# =============================================================================


class LTX2AudioEncoder(nn.Module):
    """Public wrapper for Audio Encoder with native FastVideo implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.model: AudioEncoder = AudioEncoderConfigurator.from_config(config)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(spectrogram)


class LTX2AudioDecoder(nn.Module):
    """Public wrapper for Audio Decoder with native FastVideo implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.model: AudioDecoder = AudioDecoderConfigurator.from_config(config)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.model(sample)


class LTX2Vocoder(nn.Module):
    """Public wrapper for Vocoder with native FastVideo implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.model: Vocoder = VocoderConfigurator.from_config(config)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(spectrogram)


def decode_audio(
    latent: torch.Tensor, audio_decoder: AudioDecoder, vocoder: Vocoder
) -> torch.Tensor:
    """
    Decode an audio latent representation using the provided audio decoder and vocoder.
    """
    decoded_audio = audio_decoder(latent)
    decoded_audio = vocoder(decoded_audio).squeeze(0).float()
    return decoded_audio
