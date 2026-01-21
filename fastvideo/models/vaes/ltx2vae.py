# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Video VAE implementation
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterator, List, NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastvideo.models.vaes.common import DiagonalGaussianDistribution


# =============================================================================
# Enums
# =============================================================================


class NormLayerType(Enum):
    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


class LogVarianceType(Enum):
    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


# =============================================================================
# Tiling Data Structures
# =============================================================================


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for dividing each frame into spatial tiles with optional overlap."""
    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}")
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}")
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}")
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Configuration for dividing a video into temporal tiles (chunks of frames) with optional overlap."""
    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}")
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}")
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}")
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for splitting video into tiles with optional overlap."""
    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24),
        )


@dataclass(frozen=True)
class DimensionIntervals:
    """Intervals which a single dimension of the latent space is split into."""
    starts: List[int]
    ends: List[int]
    left_ramps: List[int]
    right_ramps: List[int]


@dataclass(frozen=True)
class LatentIntervals:
    """Intervals which the latent tensor of given shape is split into."""
    original_shape: torch.Size
    dimension_intervals: Tuple[DimensionIntervals, ...]


class VideoLatentShape(NamedTuple):
    """Shape of the tensor representing video in VAE latent space."""
    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    def upscale(self, time_scale: int, spatial_scale: int) -> "VideoLatentShape":
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * time_scale + 1,
            height=self.height * spatial_scale,
            width=self.width * spatial_scale,
        )


# Operation to split a single dimension of the tensor into intervals based on the length along the dimension.
SplitOperation = Callable[[int], DimensionIntervals]
# Operation to map the intervals in input dimension to slices and masks along a corresponding output dimension.
MappingOperation = Callable[[DimensionIntervals], Tuple[List[slice], List[torch.Tensor | None]]]


class Tile(NamedTuple):
    """Represents a single tile."""
    in_coords: Tuple[slice, ...]
    out_coords: Tuple[slice, ...]
    masks_1d: Tuple[torch.Tensor | None, ...]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: List[torch.Tensor] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            if mask_1d is None:
                one = torch.ones(1)
                view_shape[dim_idx] = 1
                per_dimension_masks.append(one.view(*view_shape))
                continue

            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))

        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


# =============================================================================
# Tiling Helper Functions
# =============================================================================


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> torch.Tensor:
    """Generate a 1D trapezoidal blending mask with linear ramps."""
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = torch.ones(length)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in

    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out

    return mask.clamp_(0, 1)


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(starts=[0], ends=[length], left_ramps=[0], right_ramps=[0])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def default_mapping_operation(
    _intervals: DimensionIntervals,
) -> Tuple[List[slice], List[torch.Tensor | None]]:
    return [slice(0, None)], [None]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


def split_in_spatial(size: int, overlap: int) -> SplitOperation:
    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        return DimensionIntervals(starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps)

    return split


def split_in_temporal(size: int, overlap: int) -> SplitOperation:
    non_causal_split = split_in_spatial(size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)
        starts = list(intervals.starts)
        starts[1:] = [s - 1 for s in starts[1:]]
        left_ramps = list(intervals.left_ramps)
        left_ramps[1:] = [r + 1 for r in left_ramps[1:]]
        return replace(intervals, starts=starts, left_ramps=left_ramps)

    return split


def map_temporal_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp_scaled = 1 + (left_ramp - 1) * scale
    right_ramp_scaled = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp_scaled, right_ramp_scaled, True)


def map_spatial_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = end * scale
    left_ramp_scaled = left_ramp * scale
    right_ramp_scaled = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp_scaled, right_ramp_scaled, False)


def to_mapping_operation(
    map_func: Callable[[int, int, int, int, int], Tuple[slice, torch.Tensor]],
    scale: int,
) -> MappingOperation:
    def map_op(intervals: DimensionIntervals) -> Tuple[List[slice], List[torch.Tensor | None]]:
        output_slices: List[slice] = []
        masks_1d: List[torch.Tensor | None] = []
        number_of_slices = len(intervals.starts)
        for i in range(number_of_slices):
            start = intervals.starts[i]
            end = intervals.ends[i]
            left_ramp = intervals.left_ramps[i]
            right_ramp = intervals.right_ramps[i]
            output_slice, mask_1d = map_func(start, end, left_ramp, right_ramp, scale)
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def create_tiles_from_intervals_and_mappers(
    intervals: LatentIntervals,
    mappers: List[MappingOperation],
) -> List[Tile]:
    full_dim_input_slices = []
    full_dim_output_slices = []
    full_dim_masks_1d = []
    for axis_index in range(len(intervals.original_shape)):
        dimension_intervals = intervals.dimension_intervals[axis_index]
        starts = dimension_intervals.starts
        ends = dimension_intervals.ends
        input_slices = [slice(s, e) for s, e in zip(starts, ends, strict=True)]
        output_slices, masks_1d = mappers[axis_index](dimension_intervals)
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    tiles = []
    tile_in_coords = list(itertools.product(*full_dim_input_slices))
    tile_out_coords = list(itertools.product(*full_dim_output_slices))
    tile_mask_1ds = list(itertools.product(*full_dim_masks_1d))
    for in_coord, out_coord, mask_1d in zip(tile_in_coords, tile_out_coords, tile_mask_1ds, strict=True):
        tiles.append(
            Tile(
                in_coords=in_coord,
                out_coords=out_coord,
                masks_1d=mask_1d,
            )
        )
    return tiles


def create_tiles(
    latent_shape: torch.Size,
    splitters: List[SplitOperation],
    mappers: List[MappingOperation],
) -> List[Tile]:
    if len(splitters) != len(latent_shape):
        raise ValueError(
            f"Number of splitters must be equal to number of dimensions in latent shape, "
            f"got {len(splitters)} and {len(latent_shape)}"
        )
    if len(mappers) != len(latent_shape):
        raise ValueError(
            f"Number of mappers must be equal to number of dimensions in latent shape, "
            f"got {len(mappers)} and {len(latent_shape)}"
        )
    intervals = [splitter(length) for splitter, length in zip(splitters, latent_shape, strict=True)]
    latent_intervals = LatentIntervals(original_shape=latent_shape, dimension_intervals=tuple(intervals))
    return create_tiles_from_intervals_and_mappers(latent_intervals, mappers)


# =============================================================================
# Utility Functions
# =============================================================================


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange spatial dimensions into channels (space-to-depth).
    
    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width.
        patch_size_t: Temporal patch size. Default=1 (no temporal patching).
    
    For 5D: (B, C, F, H, W) -> (B, C*patch_size_hw^2*patch_size_t, F/patch_size_t, H/patch_size_hw, W/patch_size_hw)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange channels back into spatial dimensions (depth-to-space).
    
    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width.
        patch_size_t: Temporal patch size. Default=1 (no temporal expansion).
    
    For 5D: (B, C*patch_size_hw^2*patch_size_t, F, H, W) -> (B, C, F*patch_size_t, H*patch_size_hw, W*patch_size_hw)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    return x


# =============================================================================
# Normalization Layers
# =============================================================================


class PixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.
    Normalizes along the channel dimension using root-mean-square.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        return x / torch.sqrt(mean_sq + self.eps)


# =============================================================================
# Per-Channel Statistics for Latent Normalization
# =============================================================================


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.
    Statistics are computed over the dataset and stored in the model checkpoint.
    """

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))
        self.register_buffer("mean-of-stds", torch.ones(latent_channels))
        self.register_buffer("mean-of-stds_over_std-of-means", torch.ones(latent_channels))
        self.register_buffer("channel", torch.arange(latent_channels, dtype=torch.float32))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        return x * std + mean

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        return (x - mean) / std


# =============================================================================
# Causal 3D Convolution
# =============================================================================


class CausalConv3d(nn.Module):
    """
    Causal 3D convolution that pads temporally by repeating the first frame.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | Tuple[int, int, int] = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel_size[0]

        dilation = (dilation, 1, 1)

        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=spatial_padding_mode.value,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if causal:
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
            x = torch.cat((first_frame_pad, x), dim=2)
        else:
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
            last_frame_pad = x[:, :, -1:, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
            x = torch.cat((first_frame_pad, x, last_frame_pad), dim=2)
        x = self.conv(x)
        return x

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight


def make_conv_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int | Tuple[int, int, int] = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causal: bool = False,
    spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
) -> nn.Module:
    """Create a convolution layer (2D or 3D, causal or not)."""
    if dims == 2:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                spatial_padding_mode=spatial_padding_mode,
            )
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def make_linear_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    bias: bool = True,
) -> nn.Module:
    """Create a 1x1 convolution (pointwise linear)."""
    if dims == 2:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    elif dims == 3:
        return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


# =============================================================================
# ResNet Blocks
# =============================================================================


class ResnetBlock3D(nn.Module):
    """A 3D ResNet block with optional timestep conditioning and noise injection."""

    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise

        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.norm1 = PixelNorm()

        self.non_linearity = nn.SiLU()

        self.conv1 = make_conv_nd(
            dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.norm2 = PixelNorm()

        self.dropout = nn.Dropout(dropout)

        self.conv2 = make_conv_nd(
            dims,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale2 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        self.conv_shortcut = (
            make_linear_nd(dims=dims, in_channels=in_channels, out_channels=out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.norm3 = (
            nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=eps, affine=True)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

    def _feed_spatial_noise(
        self,
        hidden_states: torch.Tensor,
        per_channel_scale: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        spatial_shape = hidden_states.shape[-2:]
        device = hidden_states.device
        dtype = hidden_states.dtype

        spatial_noise = torch.randn(spatial_shape, device=device, dtype=dtype, generator=generator)[None]
        scaled_noise = (spatial_noise * per_channel_scale)[None, :, None, ...]
        hidden_states = hidden_states + scaled_noise
        return hidden_states

    def forward(
        self,
        input_tensor: torch.Tensor,
        causal: bool = True,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        hidden_states = input_tensor
        batch_size = hidden_states.shape[0]

        hidden_states = self.norm1(hidden_states)
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' must be provided when 'timestep_conditioning' is True")
            ada_values = self.scale_shift_table[None, ..., None, None, None].to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ) + timestep.reshape(
                batch_size,
                4,
                -1,
                timestep.shape[-3],
                timestep.shape[-2],
                timestep.shape[-1],
            )
            shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale1) + shift1

        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv1(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale1.to(device=hidden_states.device, dtype=hidden_states.dtype),
                generator=generator,
            )

        hidden_states = self.norm2(hidden_states)

        if self.timestep_conditioning:
            hidden_states = hidden_states * (1 + scale2) + shift2

        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale2.to(device=hidden_states.device, dtype=hidden_states.dtype),
                generator=generator,
            )

        input_tensor = self.norm3(input_tensor)
        input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = input_tensor + hidden_states
        return output_tensor


class UNetMidBlock3D(nn.Module):
    """A 3D UNet mid-block with multiple residual blocks."""

    def __init__(
        self,
        dims: int,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: NormLayerType = NormLayerType.GROUP_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
        attention_head_dim: int | None = None,  # unused, for compatibility
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=in_channels * 4, size_emb_dim=0
            )

        self.res_blocks = nn.ModuleList(
            [
                ResnetBlock3D(
                    dims=dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        timestep_embed = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' must be provided when 'timestep_conditioning' is True")
            batch_size = hidden_states.shape[0]
            timestep_embed = self.time_embedder(
                timestep=timestep.flatten(),
                hidden_dtype=hidden_states.dtype,
            )
            timestep_embed = timestep_embed.view(batch_size, timestep_embed.shape[-1], 1, 1, 1)

        for resnet in self.res_blocks:
            hidden_states = resnet(
                hidden_states,
                causal=causal,
                timestep=timestep_embed,
                generator=generator,
            )

        return hidden_states


# =============================================================================
# Timestep Embedding (simplified version for decoder conditioning)
# =============================================================================


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """Timestep embeddings for decoder conditioning."""

    def __init__(self, embedding_dim: int, size_emb_dim: int = 0):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        return timesteps_emb


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """MLP for timestep embeddings."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


# =============================================================================
# Sampling Blocks (Downsampling / Upsampling)
# =============================================================================


class SpaceToDepthDownsample(nn.Module):
    """Downsampling via space-to-depth with residual connection."""

    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int],
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels // math.prod(stride),
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if self.stride[0] == 2:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)

        x_in = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size)
        x_in = x_in.mean(dim=2)

        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )

        x = x + x_in
        return x


class DepthToSpaceUpsample(nn.Module):
    """Upsampling via depth-to-space (pixel shuffle)."""

    def __init__(
        self,
        dims: int,
        in_channels: int,
        stride: Tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if self.residual:
            x_in = rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.stride[0],
                p2=self.stride[1],
                p3=self.stride[2],
            )
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]

        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]
        if self.residual:
            x = x + x_in
        return x


# =============================================================================
# Block Factory Functions
# =============================================================================


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """Create an encoder block based on the block name and config."""
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown block: {block_name}")

    return block, out_channels


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """Create a decoder block based on the block name and config."""
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown layer: {block_name}")

    return block, out_channels


# =============================================================================
# Video Encoder
# =============================================================================


class VideoEncoder(nn.Module):
    """
    LTX-2 Video Encoder. Encodes video frames into a latent representation.
    """

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: list[tuple[str, int]] | list[tuple[str, dict[str, Any]]] = [],
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        latent_log_var: LogVarianceType = LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        in_channels = in_channels * patch_size**2
        feature_channels = out_channels

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in encoder_blocks:
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )

            self.down_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1
        elif latent_log_var != LogVarianceType.NONE:
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                "(e.g., 1, 9, 17, ...). Please check your input."
            )

        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == LogVarianceType.UNIFORM:
            if sample.shape[1] < 2:
                raise ValueError(
                    f"Invalid channel count for UNIFORM mode: expected at least 2, got {sample.shape[1]}"
                )
            means = sample[:, :-1, ...]
            logvar = sample[:, -1:, ...]
            num_channels = means.shape[1]
            repeat_shape = [1, num_channels] + [1] * (sample.ndim - 2)
            repeated_logvar = logvar.repeat(*repeat_shape)
            sample = torch.cat([means, repeated_logvar], dim=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        means, _ = torch.chunk(sample, 2, dim=1)
        return self.per_channel_statistics.normalize(means)


# =============================================================================
# Video Decoder
# =============================================================================


class VideoDecoder(nn.Module):
    """
    LTX-2 Video Decoder. Decodes latent representation into video frames.
    """

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: list[tuple[str, int | dict]] = [],
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        super().__init__()

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )

            self.up_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=feature_channels * 2, size_emb_dim=0
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = sample.shape[0]

        if self.timestep_conditioning:
            noise = (
                torch.randn(
                    sample.size(),
                    generator=generator,
                    dtype=sample.dtype,
                    device=sample.device,
                )
                * self.decode_noise_scale
            )
            sample = noise + (1.0 - self.decode_noise_scale) * sample

        sample = self.per_channel_statistics.un_normalize(sample)

        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype)

        sample = self.conv_in(sample, causal=self.causal)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)

        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                block_kwargs = {
                    "causal": self.causal,
                    "timestep": scaled_timestep if self.timestep_conditioning else None,
                    "generator": generator,
                }
                sample = up_block(sample, **block_kwargs)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample


# =============================================================================
# Configurators (for loading from config dict)
# =============================================================================


class VideoEncoderConfigurator:
    """Configurator for creating a video VAE Encoder from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: dict) -> VideoEncoder:
        config = config.get("vae", config)
        convolution_dimensions = config.get("dims", 3)
        in_channels = config.get("in_channels", 3)
        latent_channels = config.get("latent_channels", 128)
        encoder_spatial_padding_mode = PaddingModeType(config.get("encoder_spatial_padding_mode", "zeros"))
        encoder_blocks = config.get("encoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        latent_log_var_str = config.get("latent_log_var", "uniform")

        return VideoEncoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=latent_channels,
            encoder_blocks=encoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            latent_log_var=LogVarianceType(latent_log_var_str),
            encoder_spatial_padding_mode=encoder_spatial_padding_mode,
        )


class VideoDecoderConfigurator:
    """Configurator for creating a video VAE Decoder from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: dict) -> VideoDecoder:
        config = config.get("vae", config)
        convolution_dimensions = config.get("dims", 3)
        latent_channels = config.get("latent_channels", 128)
        decoder_spatial_padding_mode = PaddingModeType(config.get("decoder_spatial_padding_mode", "reflect"))
        out_channels = config.get("out_channels", 3)
        decoder_blocks = config.get("decoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        causal = config.get("causal_decoder", False)
        timestep_conditioning = config.get("timestep_conditioning", True)

        return VideoDecoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=latent_channels,
            out_channels=out_channels,
            decoder_blocks=decoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            decoder_spatial_padding_mode=decoder_spatial_padding_mode,
        )


# =============================================================================
# Public API (Wrapper Classes)
# =============================================================================


class LTX2VideoEncoder(nn.Module):
    """LTX-2 Video Encoder wrapper for FastVideo compatibility."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.model: VideoEncoder = VideoEncoderConfigurator.from_config(config)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.model(sample)


class LTX2VideoDecoder(nn.Module):
    """LTX-2 Video Decoder wrapper for FastVideo compatibility."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.model: VideoDecoder = VideoDecoderConfigurator.from_config(config)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.model(sample, timestep=timestep, generator=generator)


class LTX2CausalVideoAutoencoder(nn.Module):
    """
    LTX-2 VAE that exposes FastVideo's VAE encode/decode interface.
    Supports tiled decoding to reduce memory usage for high-resolution videos.
    """

    # LTX-2 VAE scale factors
    TIME_SCALE: int = 8
    SPATIAL_SCALE: int = 32

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.encoder = VideoEncoderConfigurator.from_config(config)
        self.decoder = VideoDecoderConfigurator.from_config(config)
        self._use_tiling: bool = False

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        means = self.encoder(x)
        zeros = torch.zeros_like(means)
        return DiagonalGaussianDistribution(torch.cat([means, zeros], dim=1), deterministic=True)

    def decode(
        self,
        z: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Decode latents to video, using tiling if enabled."""
        if self._use_tiling:
            # Collect all chunks from tiled decode and concatenate
            chunks = list(self.tiled_decode(z, TilingConfig.default(), timestep, generator))
            return torch.cat(chunks, dim=2)  # Concatenate along temporal dimension
        return self.decoder(z, timestep=timestep, generator=generator)

    def enable_tiling(self) -> None:
        """Enable tiled decoding with default configuration."""
        self._use_tiling = True

    def disable_tiling(self) -> None:
        """Disable tiled decoding."""
        self._use_tiling = False

    def _prepare_tiles(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None,
    ) -> List[Tile]:
        """Prepare tiles for tiled decoding based on tiling configuration."""
        splitters: List[SplitOperation] = [DEFAULT_SPLIT_OPERATION] * 5
        mappers: List[MappingOperation] = [DEFAULT_MAPPING_OPERATION] * 5

        if tiling_config is not None and tiling_config.spatial_config is not None:
            cfg = tiling_config.spatial_config
            tile_size = cfg.tile_size_in_pixels // self.SPATIAL_SCALE
            overlap = cfg.tile_overlap_in_pixels // self.SPATIAL_SCALE
            splitters[3] = split_in_spatial(tile_size, overlap)
            splitters[4] = split_in_spatial(tile_size, overlap)
            mappers[3] = to_mapping_operation(map_spatial_slice, self.SPATIAL_SCALE)
            mappers[4] = to_mapping_operation(map_spatial_slice, self.SPATIAL_SCALE)

        if tiling_config is not None and tiling_config.temporal_config is not None:
            cfg = tiling_config.temporal_config
            tile_size = cfg.tile_size_in_frames // self.TIME_SCALE
            overlap = cfg.tile_overlap_in_frames // self.TIME_SCALE
            splitters[2] = split_in_temporal(tile_size, overlap)
            mappers[2] = to_mapping_operation(map_temporal_slice, self.TIME_SCALE)

        return create_tiles(latent.shape, splitters, mappers)

    def _group_tiles_by_temporal_slice(self, tiles: List[Tile]) -> List[List[Tile]]:
        """Group tiles by their temporal output slice."""
        if not tiles:
            return []

        groups = []
        current_slice = tiles[0].out_coords[2]
        current_group = []

        for tile in tiles:
            tile_slice = tile.out_coords[2]
            if tile_slice == current_slice:
                current_group.append(tile)
            else:
                groups.append(current_group)
                current_slice = tile_slice
                current_group = [tile]

        if current_group:
            groups.append(current_group)

        return groups

    def _accumulate_temporal_group_into_buffer(
        self,
        group_tiles: List[Tile],
        buffer: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Decode and accumulate all tiles of a temporal group into a local buffer."""
        temporal_slice = group_tiles[0].out_coords[2]
        weights = torch.zeros_like(buffer)

        for tile in group_tiles:
            decoded_tile = self.decoder(latent[tile.in_coords], timestep, generator)
            mask = tile.blend_mask.to(device=buffer.device, dtype=buffer.dtype)
            temporal_offset = tile.out_coords[2].start - temporal_slice.start
            expected_temporal_len = tile.out_coords[2].stop - tile.out_coords[2].start
            decoded_temporal_len = decoded_tile.shape[2]

            actual_temporal_len = min(expected_temporal_len, decoded_temporal_len, buffer.shape[2] - temporal_offset)

            chunk_coords = (
                slice(None),  # batch
                slice(None),  # channels
                slice(temporal_offset, temporal_offset + actual_temporal_len),
                tile.out_coords[3],  # height
                tile.out_coords[4],  # width
            )

            decoded_slice = decoded_tile[:, :, :actual_temporal_len, :, :]
            mask_slice = mask[:, :, :actual_temporal_len, :, :] if mask.shape[2] > 1 else mask

            buffer[chunk_coords] += decoded_slice * mask_slice
            weights[chunk_coords] += mask_slice

        return weights

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """
        Decode a latent tensor into video frames using tiled processing.
        Splits the latent tensor into tiles, decodes each tile individually,
        and yields video chunks as they become available.
        
        Args:
            latent: Input latent tensor (B, C, F', H', W').
            tiling_config: Tiling configuration for the latent tensor.
            timestep: Optional timestep for decoder conditioning.
            generator: Optional random generator for deterministic decoding.
            
        Yields:
            Video chunks (B, C, T, H, W) by temporal slices.
        """
        full_video_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(
            self.TIME_SCALE, self.SPATIAL_SCALE
        )
        tiles = self._prepare_tiles(latent, tiling_config)
        temporal_groups = self._group_tiles_by_temporal_slice(tiles)

        previous_chunk = None
        previous_weights = None
        previous_temporal_slice = None

        for temporal_group_tiles in temporal_groups:
            curr_temporal_slice = temporal_group_tiles[0].out_coords[2]

            temporal_tile_buffer_shape = full_video_shape._replace(
                frames=curr_temporal_slice.stop - curr_temporal_slice.start,
            )

            buffer = torch.zeros(
                temporal_tile_buffer_shape.to_torch_shape(),
                device=latent.device,
                dtype=latent.dtype,
            )

            curr_weights = self._accumulate_temporal_group_into_buffer(
                group_tiles=temporal_group_tiles,
                buffer=buffer,
                latent=latent,
                timestep=timestep,
                generator=generator,
            )

            # Blend with previous temporal chunk if it exists
            if previous_chunk is not None:
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    temporal_overlap_slice = slice(curr_temporal_slice.start - previous_temporal_slice.start, None)

                    previous_chunk[:, :, temporal_overlap_slice, :, :] += buffer[:, :, slice(0, overlap_len), :, :]
                    previous_weights[:, :, temporal_overlap_slice, :, :] += curr_weights[
                        :, :, slice(0, overlap_len), :, :
                    ]

                    buffer[:, :, slice(0, overlap_len), :, :] = previous_chunk[:, :, temporal_overlap_slice, :, :]
                    curr_weights[:, :, slice(0, overlap_len), :, :] = previous_weights[
                        :, :, temporal_overlap_slice, :, :
                    ]

                # Yield the non-overlapping part of the previous chunk
                previous_weights = previous_weights.clamp(min=1e-8)
                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                yield (previous_chunk / previous_weights)[:, :, :yield_len, :, :]

            # Update state for next iteration
            previous_chunk = buffer
            previous_weights = curr_weights
            previous_temporal_slice = curr_temporal_slice

        # Yield any remaining chunk
        if previous_chunk is not None:
            previous_weights = previous_weights.clamp(min=1e-8)
            yield previous_chunk / previous_weights
