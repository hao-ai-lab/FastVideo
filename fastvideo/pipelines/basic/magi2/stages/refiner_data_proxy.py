# Copyright (c) 2026 SandAI. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack MAGI-2 refiner video, audio, text, and reference tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

import torch
from einops import rearrange
from torch.nn import functional as F


class Modality(IntEnum):
    """Modality identifiers consumed by the MAGI-2 refiner transformer."""

    VIDEO = 0
    AUDIO = 1
    TEXT = 2


@dataclass
class VarlenHandler:
    """Variable-length attention boundaries for a packed refiner sequence."""

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int | torch.Tensor
    max_seqlen_k: int | torch.Tensor


@dataclass
class WindowLocalAttnHandler:
    """Query and key ranges consumed by MAGI-2 window attention."""

    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    attn_type_map: torch.Tensor
    softmax_scale: float | None = None
    bwd_q_ranges: torch.Tensor | None = None
    bwd_k_ranges: torch.Tensor | None = None
    bwd_attn_type_map: torch.Tensor | None = None
    auto_range_merge: bool = False
    sparse_load: bool = False


@dataclass(frozen=True)
class BlockLocalAttnRanges:
    """Forward and backward attention ranges for video blocks."""

    fwd_q_ranges: torch.Tensor
    fwd_k_ranges: torch.Tensor
    bwd_q_ranges: torch.Tensor
    bwd_k_ranges: torch.Tensor
    max_q_len: int


@dataclass(frozen=True)
class Magi2RefinerDataProxyConfig:
    """Configuration that controls MAGI-2 refiner token packing."""

    t_patch_size: int = 1
    patch_size: int = 2
    frame_receptive_field: int = 11
    spatial_rope_interpolation: Literal["inter", "extra"] = "extra"
    text_offset: int = 0
    coords_style: Literal["v1", "v2"] = "v1"
    attn_config: dict[str, Any] = field(default_factory=dict)
    magi2_refiner_condition_input: Literal["none", "concat_x1"] = "none"


@dataclass
class RefinerModelInput:
    """Multimodal tensors and lengths consumed by the refiner data proxy."""

    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor
    ref_audio_feat: torch.Tensor
    ref_audio_feat_len: torch.Tensor
    ref_video_feat: torch.Tensor
    ref_video_feat_len: torch.Tensor


def _to_int(value: int | torch.Tensor) -> int:
    """Convert a scalar tensor or Python integer into an integer."""
    if isinstance(value, torch.Tensor):
        return int(value.detach().reshape(-1)[0].item())
    return int(value)


def _max_range_len(ranges: torch.Tensor) -> int:
    """Return the longest query range in a two-column range tensor."""
    if ranges.numel() == 0:
        return 0
    lengths = ranges[:, 1].to(torch.int64) - ranges[:, 0].to(torch.int64)
    return int(lengths.max().item())


def _cat_ranges(
    *ranges: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Concatenate non-empty attention ranges as contiguous int32 rows."""
    valid_ranges = [value for value in ranges if value is not None and value.numel() > 0]
    if valid_ranges:
        return torch.cat(valid_ranges, dim=0).to(dtype=torch.int32).contiguous()
    return torch.empty((0, 2), device=device, dtype=torch.int32)


class BlockLocalAttn:
    """Build scan-window attention ranges over an ordered list of blocks."""

    def __init__(self, block_ranges_tensor: torch.Tensor, win_size: int) -> None:
        self.block_ranges_tensor = block_ranges_tensor.to(dtype=torch.int32).contiguous()
        self.win_size = int(win_size)

    @property
    def num_blocks(self) -> int:
        """Return the number of blocks in the scan order."""
        return int(self.block_ranges_tensor.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the device that owns the block ranges."""
        return self.block_ranges_tensor.device

    def build_ranges(self) -> BlockLocalAttnRanges:
        """Build centered scan windows and their backward-attention ranges."""
        if self.num_blocks == 0:
            empty = torch.empty((0, 2), device=self.device, dtype=torch.int32)
            return BlockLocalAttnRanges(empty, empty, empty, empty, 0)
        window_width = min(self.num_blocks, max(1, self.win_size))
        query_block = torch.arange(self.num_blocks, device=self.device, dtype=torch.int32)
        left_width = window_width // 2
        window_start = (query_block - left_width).clamp(
            min=0,
            max=self.num_blocks - window_width,
        )
        window_end = window_start + window_width
        query_ranges = self.block_ranges_tensor[query_block.long()].contiguous()
        key_ranges = torch.stack(
            [
                self.block_ranges_tensor[window_start.long(), 0],
                self.block_ranges_tensor[(window_end - 1).long(), 1],
            ],
            dim=1,
        ).contiguous()

        key_block = torch.arange(self.num_blocks, device=self.device, dtype=torch.int32)
        first_query_block = torch.searchsorted(window_end, key_block, right=True, out_int32=True)
        last_query_block = torch.searchsorted(window_start, key_block, right=True, out_int32=True) - 1
        backward_query_ranges = torch.stack(
            [
                self.block_ranges_tensor[first_query_block.long(), 0],
                self.block_ranges_tensor[last_query_block.long(), 1],
            ],
            dim=1,
        ).contiguous()
        backward_key_ranges = self.block_ranges_tensor[key_block.long()].contiguous()
        return BlockLocalAttnRanges(
            query_ranges,
            key_ranges,
            backward_query_ranges,
            backward_key_ranges,
            _max_range_len(query_ranges),
        )


class BlockGridLocalAttn:
    """Build radius-based attention ranges over a three-dimensional block grid."""

    def __init__(
        self,
        block_ranges_tensor: torch.Tensor,
        block_grid_shape: tuple[int, int, int],
        radius: tuple[int, int, int],
    ) -> None:
        """Validate and store the block grid used to construct local ranges."""
        self.block_ranges_tensor = block_ranges_tensor.to(dtype=torch.int32).contiguous()
        self.block_grid_shape = tuple(int(dimension) for dimension in block_grid_shape)
        self.radius = tuple(int(value) for value in radius)
        expected_num_blocks = math.prod(self.block_grid_shape)
        if expected_num_blocks != int(self.block_ranges_tensor.shape[0]):
            raise ValueError(
                "block_grid_shape product must match block count, got "
                f"expected_num_blocks={expected_num_blocks} "
                f"num_blocks={self.block_ranges_tensor.shape[0]}"
            )

    @property
    def num_blocks(self) -> int:
        """Return the number of blocks in the spatial-temporal grid."""
        return int(self.block_ranges_tensor.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the device that owns the block ranges."""
        return self.block_ranges_tensor.device

    def build_ranges(self) -> BlockLocalAttnRanges:
        """Pair each query block with every in-bounds neighbor in its radius."""
        if self.num_blocks == 0:
            empty = torch.empty((0, 2), device=self.device, dtype=torch.int32)
            return BlockLocalAttnRanges(empty, empty, empty, empty, 0)
        time_blocks, height_blocks, width_blocks = self.block_grid_shape
        time_radius, height_radius, width_radius = self.radius
        query_block = torch.arange(self.num_blocks, device=self.device, dtype=torch.int64)
        query_time = query_block // (height_blocks * width_blocks)
        query_height = (query_block // width_blocks) % height_blocks
        query_width = query_block % width_blocks
        query_ranges_by_block = self.block_ranges_tensor[query_block]
        query_range_parts: list[torch.Tensor] = []
        key_range_parts: list[torch.Tensor] = []
        for time_delta in range(-time_radius, time_radius + 1):
            key_time = query_time + time_delta
            valid_time = (key_time >= 0) & (key_time < time_blocks)
            if not bool(valid_time.any()):
                continue
            for height_delta in range(-height_radius, height_radius + 1):
                key_height = query_height + height_delta
                valid_time_height = valid_time & (key_height >= 0) & (key_height < height_blocks)
                if not bool(valid_time_height.any()):
                    continue
                for width_delta in range(-width_radius, width_radius + 1):
                    key_width = query_width + width_delta
                    valid = valid_time_height & (key_width >= 0) & (key_width < width_blocks)
                    if not bool(valid.any()):
                        continue
                    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
                    key_block = key_time * (height_blocks * width_blocks) + key_height * width_blocks + key_width
                    query_range_parts.append(query_ranges_by_block[valid_indices])
                    key_range_parts.append(self.block_ranges_tensor[key_block[valid_indices]])
        if not query_range_parts:
            empty = torch.empty((0, 2), device=self.device, dtype=torch.int32)
            return BlockLocalAttnRanges(empty, empty, empty, empty, 0)
        query_ranges = torch.cat(query_range_parts, dim=0).to(dtype=torch.int32).contiguous()
        key_ranges = torch.cat(key_range_parts, dim=0).to(dtype=torch.int32).contiguous()
        return BlockLocalAttnRanges(
            query_ranges,
            key_ranges,
            query_ranges,
            key_ranges,
            _max_range_len(query_ranges),
        )


def _get_coords(
    shape: tuple[int, int, int],
    ref_feat_shape: tuple[int, int, int],
    offset_thw: tuple[int, int, int] = (0, 0, 0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build token coordinates with source-grid and reference-grid metadata."""
    if device is None:
        device = torch.device("cpu")
    original_time, original_height, original_width = shape
    reference_time, reference_height, reference_width = ref_feat_shape
    time_offset, height_offset, width_offset = offset_thw
    time_range = torch.arange(original_time, device=device, dtype=dtype) + time_offset
    height_range = torch.arange(original_height, device=device, dtype=dtype) + height_offset
    width_range = torch.arange(original_width, device=device, dtype=dtype) + width_offset
    time_grid, height_grid, width_grid = torch.meshgrid(
        time_range,
        height_range,
        width_range,
        indexing="ij",
    )
    coordinates = torch.stack([time_grid, height_grid, width_grid], dim=-1).reshape(-1, 3)
    metadata = torch.tensor(
        [
            original_time,
            original_height,
            original_width,
            reference_time,
            reference_height,
            reference_width,
        ],
        device=device,
        dtype=dtype,
    )
    return torch.cat([coordinates, metadata.expand(coordinates.size(0), -1)], dim=-1)


@dataclass
class SingleData:
    """One refiner sample's token segments and packing metadata."""

    video_x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: int
    ref_audio_feat: torch.Tensor
    ref_audio_feat_len: int
    ref_video_feat: torch.Tensor
    ref_video_feat_len: int
    txt_feat: torch.Tensor
    txt_feat_len: int
    t: int
    h: int
    w: int
    patch_size: int
    t_patch_size: int
    spatial_rope_interpolation: Literal["inter", "extra"]
    text_offset: int
    coords_style: Literal["v1", "v2"] = "v1"

    def __post_init__(self) -> None:
        """Trim variable-length segments and cache channel and token counts."""
        self.audio_feat_len = _to_int(self.audio_feat_len)
        self.ref_audio_feat_len = _to_int(self.ref_audio_feat_len)
        self.ref_video_feat_len = _to_int(self.ref_video_feat_len)
        self.txt_feat_len = _to_int(self.txt_feat_len)
        self.video_token_num = self.video_x_t.shape[0]
        self.ref_video_feat = self.ref_video_feat[:self.ref_video_feat_len]
        self.audio_x_t = self.audio_x_t[:self.audio_feat_len]
        self.ref_audio_feat = self.ref_audio_feat[:self.ref_audio_feat_len]
        self.txt_feat = self.txt_feat[:self.txt_feat_len]
        self.video_channel = self.video_x_t.shape[-1]
        self.audio_channel = self.audio_x_t.shape[-1]
        self.txt_channel = self.txt_feat.shape[-1]

    @property
    def device(self) -> torch.device:
        """Return the device that owns this sample's packed tokens."""
        return self.video_x_t.device

    @property
    def default_dtype(self) -> torch.dtype:
        """Return the video-token dtype used for coordinates."""
        return self.video_x_t.dtype

    @property
    def total_token_num(self) -> int:
        """Return the complete token count for one refiner sample."""
        return (
            self.video_token_num
            + self.audio_feat_len
            + self.txt_feat_len
            + self.ref_audio_feat_len
            + self.ref_video_feat_len
        )

    @property
    def token_sequence(self) -> torch.Tensor:
        """Return channel-padded token segments in transformer consumption order."""
        token_segments = [
            self.video_x_t,
            self.audio_x_t,
            self.txt_feat,
            self.ref_audio_feat,
            self.ref_video_feat,
        ]
        max_channel = max(segment.shape[-1] for segment in token_segments)
        padded_segments = [
            F.pad(segment, (0, max_channel - segment.shape[-1]))
            for segment in token_segments
        ]
        return torch.cat(padded_segments, dim=0)

    @property
    def modality_mapping(self) -> torch.Tensor:
        """Return one modality identifier per token in transformer order."""
        video_mapping = torch.full(
            (self.video_token_num,),
            int(Modality.VIDEO),
            dtype=torch.int64,
            device=self.device,
        )
        audio_mapping = torch.full(
            (self.audio_feat_len,),
            int(Modality.AUDIO),
            dtype=torch.int64,
            device=self.device,
        )
        text_mapping = torch.full(
            (self.txt_feat_len,),
            int(Modality.TEXT),
            dtype=torch.int64,
            device=self.device,
        )
        reference_audio_mapping = torch.full(
            (self.ref_audio_feat_len,),
            int(Modality.AUDIO),
            dtype=torch.int64,
            device=self.device,
        )
        reference_video_mapping = torch.full(
            (self.ref_video_feat_len,),
            int(Modality.VIDEO),
            dtype=torch.int64,
            device=self.device,
        )
        return torch.cat(
            [
                video_mapping,
                audio_mapping,
                text_mapping,
                reference_audio_mapping,
                reference_video_mapping,
            ],
            dim=0,
        )

    def _default_coords(
        self,
        shape: tuple[int, int, int],
        ref_feat_shape: tuple[int, int, int],
        offset_thw: tuple[int, int, int] = (0, 0, 0),
    ) -> torch.Tensor:
        """Build coordinate rows on this sample's device and dtype."""
        return _get_coords(
            shape=shape,
            ref_feat_shape=ref_feat_shape,
            offset_thw=offset_thw,
            device=self.device,
            dtype=self.default_dtype,
        )

    @property
    def coords_mapping(self) -> torch.Tensor:
        """Return source and reference coordinates for every packed token."""
        time_steps = self.t // self.t_patch_size
        height_steps = self.h // self.patch_size
        width_steps = self.w // self.patch_size
        if self.spatial_rope_interpolation == "inter":
            video_reference_shape = (time_steps, 32, 32)
        else:
            video_reference_shape = (time_steps, height_steps, width_steps)
        video_coords = self._default_coords(
            shape=(time_steps, height_steps, width_steps),
            ref_feat_shape=video_reference_shape,
        )

        reference_video_spatial_size = (
            int(math.ceil(math.sqrt(self.ref_video_feat_len)))
            if self.ref_video_feat_len > 0
            else 10
        )
        audio_reference_time = (self.audio_feat_len - 1) // 8 + 1
        audio_coords = self._default_coords(
            shape=(self.audio_feat_len, 1, 1),
            ref_feat_shape=(audio_reference_time // self.t_patch_size, 1, 1),
        )
        text_coords = self._default_coords(
            shape=(self.txt_feat_len, 1, 1),
            ref_feat_shape=(1, 1, 1),
            offset_thw=(-self.txt_feat_len, 0, 0),
        )
        reference_audio_time = math.ceil(
            ((self.ref_audio_feat_len - 1) // 8 + 1) / self.t_patch_size
        )
        if self.ref_audio_feat_len > 1:
            reference_audio_time = max(reference_audio_time, 2)
        reference_audio_coords = self._default_coords(
            shape=(self.ref_audio_feat_len, 1, 1),
            ref_feat_shape=(reference_audio_time, 1, 1),
            offset_thw=(2 * self.audio_feat_len, 0, 0),
        )
        reference_video_coords = self._default_coords(
            shape=(1, reference_video_spatial_size, reference_video_spatial_size),
            ref_feat_shape=(1, reference_video_spatial_size, reference_video_spatial_size),
            offset_thw=(1000, 0, 0),
        )[:self.ref_video_feat_len]
        return torch.cat(
            [
                video_coords,
                audio_coords,
                text_coords,
                reference_audio_coords,
                reference_video_coords,
            ],
            dim=0,
        )

    def depack_token_sequence(
        self,
        token_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore one sample's video latent and active audio-token output."""
        video_tokens = token_sequence[:self.video_token_num, :self.video_channel]
        video = rearrange(
            video_tokens,
            "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
            H=self.h // self.patch_size,
            W=self.w // self.patch_size,
            pT=self.t_patch_size,
            pH=self.patch_size,
            pW=self.patch_size,
        ).contiguous()
        audio = token_sequence[
            self.video_token_num:self.video_token_num + self.audio_feat_len,
            :self.audio_channel,
        ]
        return video, audio


@dataclass
class PackedRefinerBatch:
    """A refiner batch represented as adjacent variable-length sequences."""

    items: list[SingleData]

    @property
    def token_sequence(self) -> torch.Tensor:
        """Return every sample's tokens as one packed tensor."""
        return torch.cat([item.token_sequence for item in self.items], dim=0)

    @property
    def modality_mapping(self) -> torch.Tensor:
        """Return modality identifiers across every packed sample."""
        return torch.cat([item.modality_mapping for item in self.items], dim=0)

    @property
    def coords_mapping(self) -> torch.Tensor:
        """Return coordinate rows across every packed sample."""
        return torch.cat([item.coords_mapping for item in self.items], dim=0)

    @property
    def total_token_num(self) -> int:
        """Return the total token count across the packed batch."""
        return sum(item.total_token_num for item in self.items)

    def __getitem__(self, index: int) -> SingleData:
        """Return one packed sample descriptor."""
        return self.items[index]

    @property
    def cu_seqlen(self) -> torch.Tensor:
        """Return cumulative token boundaries for each packed sample."""
        lengths = torch.tensor([item.total_token_num for item in self.items])
        return F.pad(torch.cumsum(lengths, dim=0), (1, 0))

    @property
    def max_seqlen(self) -> int:
        """Return the longest unpadded sample sequence."""
        return max(item.total_token_num for item in self.items)

    def depack_token_sequence(
        self,
        token_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore batched video latents and active audio-token outputs."""
        video_outputs: list[torch.Tensor] = []
        audio_outputs: list[torch.Tensor] = []
        token_counts = [item.total_token_num for item in self.items]
        for item, token_slice in zip(
            self.items,
            torch.split(token_sequence, token_counts, dim=0),
            strict=True,
        ):
            video, audio = item.depack_token_sequence(token_slice)
            video_outputs.append(video)
            audio_outputs.append(audio)
        return torch.stack(video_outputs, dim=0), torch.stack(audio_outputs, dim=0)


def _build_frame_local_attn_handler(
    num_video_tokens: int,
    num_audio_and_text_tokens: int,
    num_frames: int,
    frame_receptive_field: int,
    device: torch.device,
) -> WindowLocalAttnHandler:
    """Build the refiner's legacy frame-window attention ranges."""
    tokens_per_frame = num_video_tokens // num_frames
    total_tokens = num_video_tokens + num_audio_and_text_tokens
    query_ranges: list[torch.Tensor] = []
    key_ranges: list[torch.Tensor] = []
    for frame_index in range(num_frames):
        query_ranges.append(
            torch.tensor(
                [
                    frame_index * tokens_per_frame,
                    (frame_index + 1) * tokens_per_frame,
                ]
            )
        )
        key_ranges.append(
            torch.tensor(
                [
                    (frame_index - frame_receptive_field) * tokens_per_frame,
                    (frame_index + frame_receptive_field + 1) * tokens_per_frame,
                ]
            )
        )
    local_query_ranges = torch.stack(query_ranges, dim=0)
    local_key_ranges = torch.stack(key_ranges, dim=0)
    local_key_ranges[local_key_ranges < 0] = 0
    local_key_ranges[local_key_ranges > num_video_tokens] = num_video_tokens
    video_query_range = torch.tensor([[0, num_video_tokens]])
    video_key_range = torch.tensor(
        [[num_video_tokens, num_video_tokens + num_audio_and_text_tokens]]
    )
    context_query_range = torch.tensor([[num_video_tokens, total_tokens]])
    context_key_range = torch.tensor([[0, total_tokens]])
    packed_query_ranges = torch.cat(
        [local_query_ranges, video_query_range, context_query_range],
        dim=0,
    ).to(device=device, dtype=torch.int32, non_blocking=True)
    packed_key_ranges = torch.cat(
        [local_key_ranges, video_key_range, context_key_range],
        dim=0,
    ).to(device=device, dtype=torch.int32, non_blocking=True)
    return WindowLocalAttnHandler(
        q_ranges=packed_query_ranges,
        k_ranges=packed_key_ranges,
        max_seqlen_q=total_tokens,
        max_seqlen_k=total_tokens,
        attn_type_map=torch.zeros(
            packed_query_ranges.shape[0],
            device=device,
            dtype=torch.int32,
        ),
    )


class Magi2RefinerDataProxy:
    """Pack MAGI-2 refiner inputs and restore refiner-model outputs."""

    def __init__(self, config: Magi2RefinerDataProxyConfig) -> None:
        """Initialize patch extraction and local-attention configuration."""
        self.patch_size = config.patch_size
        self.t_patch_size = config.t_patch_size
        self.frame_receptive_field = config.frame_receptive_field
        self.spatial_rope_interpolation = config.spatial_rope_interpolation
        self.text_offset = config.text_offset
        self.coords_style = config.coords_style
        self.attn_config = dict(config.attn_config or {})
        attention_mode = str(self.attn_config.get("mode", "dense")).lower()
        if attention_mode == "local":
            attention_mode = "window"
        if attention_mode not in {"dense", "window"}:
            raise ValueError(f"Unsupported refiner attention mode: {attention_mode!r}")
        self.use_window_attn = attention_mode == "window"
        self.window_attn_config = dict(self.attn_config.get("window", {}))
        self._saved_data: dict[str, Any] = {}

    def saved_for_output(self, **kwargs: Any) -> None:
        """Save packing metadata that is required to depack model outputs."""
        self._saved_data.update(kwargs)

    def get_saved_data(self, key: str) -> Any:
        """Return saved packing metadata by name."""
        return self._saved_data[key]

    def img2tokens(self, video: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping 3D patches with channel-major features."""
        patches = (
            video.unfold(2, self.t_patch_size, self.t_patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .unfold(4, self.patch_size, self.patch_size)
        )
        batch_size, _channels, time_steps, height_steps, width_steps, _, _, _ = patches.shape
        return (
            patches.permute(0, 2, 3, 4, 1, 5, 6, 7)
            .reshape(batch_size, time_steps * height_steps * width_steps, -1)
            .contiguous()
        )

    @staticmethod
    def _dimension_block_sizes(
        dimension_size: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return valid-token counts for full and edge blocks in one dimension."""
        sizes = torch.full(
            (math.ceil(dimension_size / block_size),),
            block_size,
            device=device,
            dtype=torch.int32,
        )
        sizes[-1] = dimension_size - block_size * (sizes.numel() - 1)
        return sizes

    @classmethod
    def _build_video_block_ranges(
        cls,
        patch_counts: tuple[int, int, int],
        block_sizes: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build contiguous token ranges for every spatial-temporal video block."""
        time_count, height_count, width_count = patch_counts
        time_block, height_block, width_block = block_sizes
        time_sizes = cls._dimension_block_sizes(time_count, time_block, device)
        height_sizes = cls._dimension_block_sizes(height_count, height_block, device)
        width_sizes = cls._dimension_block_sizes(width_count, width_block, device)
        block_token_counts = (
            time_sizes[:, None, None]
            * height_sizes[None, :, None]
            * width_sizes[None, None, :]
        ).flatten()
        ends = torch.cumsum(block_token_counts, dim=0)
        starts = ends - block_token_counts
        return torch.stack([starts, ends], dim=1).contiguous()

    @classmethod
    def _build_block_order(
        cls,
        *,
        patch_counts: tuple[int, int, int],
        block_sizes: tuple[int, int, int],
        valid_token_num: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        """Reorder video tokens into contiguous blocks and retain the inverse order."""
        video_token_num = math.prod(patch_counts)
        video_token_indices = torch.arange(
            video_token_num,
            device=device,
            dtype=torch.int32,
        ).view(*patch_counts)
        padded_counts = tuple(
            math.ceil(patch_count / block_size) * block_size
            for patch_count, block_size in zip(
                patch_counts,
                block_sizes,
                strict=True,
            )
        )
        padded_video_indices = torch.full(
            padded_counts,
            -1,
            dtype=torch.int32,
            device=device,
        )
        padded_video_indices[
            :patch_counts[0],
            :patch_counts[1],
            :patch_counts[2],
        ] = video_token_indices
        time_block, height_block, width_block = block_sizes
        padded_video_indices = (
            padded_video_indices.view(
                padded_counts[0] // time_block,
                time_block,
                padded_counts[1] // height_block,
                height_block,
                padded_counts[2] // width_block,
                width_block,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(-1)
        )
        video_order = padded_video_indices[padded_video_indices >= 0]
        tail_indices = torch.arange(
            video_token_num,
            valid_token_num,
            device=device,
            dtype=torch.int32,
        )
        token_order = torch.cat([video_order, tail_indices], dim=0)
        token_restore_order = torch.empty_like(token_order)
        token_restore_order[token_order] = torch.arange(
            token_order.numel(),
            device=device,
            dtype=torch.int32,
        )
        block_grid_shape = (
            math.ceil(patch_counts[0] / time_block),
            math.ceil(patch_counts[1] / height_block),
            math.ceil(patch_counts[2] / width_block),
        )
        video_block_ranges = cls._build_video_block_ranges(
            patch_counts,
            block_sizes,
            device,
        )
        return token_order, token_restore_order, video_block_ranges, block_grid_shape

    @staticmethod
    def _dense_context_query_ranges(
        query_range: tuple[int, int],
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Split a dense context query range into bounded attention rows."""
        query_start, query_end = query_range
        if query_end <= query_start:
            return torch.empty((0, 2), device=device, dtype=torch.int32)
        query_starts = torch.arange(
            query_start,
            query_end,
            max(1, block_size),
            device=device,
            dtype=torch.int32,
        )
        query_ends = torch.clamp(query_starts + max(1, block_size), max=query_end)
        return torch.stack((query_starts, query_ends), dim=1).contiguous()

    def _build_window_local_attn_handler(
        self,
        item: SingleData,
        total_token_num: int,
        device: torch.device,
    ) -> WindowLocalAttnHandler:
        """Build block- or frame-window ranges plus dense cross-modal ranges."""
        window_config = {**self.attn_config, **self.window_attn_config}
        level = str(window_config.get("level", "block")).lower()
        if level not in {"block", "frame"}:
            raise ValueError(f"Unsupported window attention level: {level!r}")
        patch_counts = (
            item.t // item.t_patch_size,
            item.h // item.patch_size,
            item.w // item.patch_size,
        )
        tokens_per_frame = patch_counts[1] * patch_counts[2]
        video_token_num = math.prod(patch_counts)
        tail_range = (video_token_num, total_token_num)
        video_range = (0, video_token_num)

        if level == "block":
            block_sizes = (
                int(window_config.get("block_t_size", self.attn_config.get("block_t_size", 1))),
                int(window_config.get("block_size", self.attn_config.get("block_size", 1))),
                int(window_config.get("block_size", self.attn_config.get("block_size", 1))),
            )
            _, _, video_block_ranges, block_grid_shape = self._build_block_order(
                patch_counts=patch_counts,
                block_sizes=block_sizes,
                valid_token_num=total_token_num,
                device=device,
            )
            block_mode = str(
                window_config.get(
                    "block_mode",
                    window_config.get("window_block_mode", "scan"),
                )
            ).lower()
            if block_mode == "grid":
                window_ranges = BlockGridLocalAttn(
                    block_ranges_tensor=video_block_ranges,
                    block_grid_shape=block_grid_shape,
                    radius=(
                        int(window_config.get("block_t_radius", 1)),
                        int(window_config.get("block_h_radius", 1)),
                        int(window_config.get("block_w_radius", 1)),
                    ),
                ).build_ranges()
            elif block_mode == "scan":
                window_ranges = BlockLocalAttn(
                    block_ranges_tensor=video_block_ranges,
                    win_size=int(window_config.get("win_size", self.attn_config.get("win_size", 1))),
                ).build_ranges()
            else:
                raise ValueError(f"Unsupported block window mode: {block_mode!r}")
            video_dense_query_ranges = video_block_ranges
            dense_context_query_block_size = max(math.prod(block_sizes), 1)
        else:
            frame_indices = torch.arange(patch_counts[0], device=device, dtype=torch.int32)
            frame_ranges = torch.stack(
                (
                    frame_indices * tokens_per_frame,
                    (frame_indices + 1) * tokens_per_frame,
                ),
                dim=1,
            ).contiguous()
            radius = int(window_config.get("frame_receptive_field", self.frame_receptive_field))
            if radius < 0:
                raise ValueError("frame-level window attention requires frame_receptive_field >= 0")
            key_start_frames = (frame_indices - radius).clamp(min=0)
            key_end_frames = (frame_indices + radius + 1).clamp(max=patch_counts[0])
            forward_key_ranges = torch.stack(
                (
                    key_start_frames * tokens_per_frame,
                    key_end_frames * tokens_per_frame,
                ),
                dim=1,
            )
            window_ranges = BlockLocalAttnRanges(
                frame_ranges,
                forward_key_ranges,
                forward_key_ranges,
                frame_ranges,
                tokens_per_frame,
            )
            video_dense_query_ranges = frame_ranges
            dense_context_query_block_size = max(tokens_per_frame, 1)

        dense_query_ranges = torch.empty((0, 2), device=device, dtype=torch.int32)
        dense_key_ranges = torch.empty((0, 2), device=device, dtype=torch.int32)
        if tail_range[0] < tail_range[1]:
            query_parts = [
                video_dense_query_ranges,
                self._dense_context_query_ranges(
                    tail_range,
                    dense_context_query_block_size,
                    device,
                ),
                self._dense_context_query_ranges(
                    tail_range,
                    dense_context_query_block_size,
                    device,
                ),
            ]
            key_parts = [
                torch.tensor(tail_range, device=device, dtype=torch.int32)
                .view(1, 2)
                .expand(video_dense_query_ranges.shape[0], 2),
                torch.tensor(video_range, device=device, dtype=torch.int32)
                .view(1, 2)
                .expand(query_parts[1].shape[0], 2),
                torch.tensor(tail_range, device=device, dtype=torch.int32)
                .view(1, 2)
                .expand(query_parts[2].shape[0], 2),
            ]
            dense_query_ranges = torch.cat(query_parts, dim=0).contiguous()
            dense_key_ranges = torch.cat(key_parts, dim=0).contiguous()

        query_ranges = _cat_ranges(
            dense_query_ranges,
            window_ranges.fwd_q_ranges,
            device=device,
        )
        key_ranges = _cat_ranges(
            dense_key_ranges,
            window_ranges.fwd_k_ranges,
            device=device,
        )
        return WindowLocalAttnHandler(
            q_ranges=query_ranges,
            k_ranges=key_ranges,
            max_seqlen_q=max(_max_range_len(query_ranges), window_ranges.max_q_len),
            max_seqlen_k=total_token_num,
            attn_type_map=torch.zeros(
                query_ranges.shape[0],
                device=device,
                dtype=torch.int32,
            ),
            bwd_q_ranges=_cat_ranges(
                dense_query_ranges,
                window_ranges.bwd_q_ranges,
                device=device,
            ),
            bwd_k_ranges=_cat_ranges(
                dense_key_ranges,
                window_ranges.bwd_k_ranges,
                device=device,
            ),
            bwd_attn_type_map=torch.zeros(
                query_ranges.shape[0],
                device=device,
                dtype=torch.int32,
            ),
            auto_range_merge=bool(window_config.get("auto_range_merge", False)),
            sparse_load=bool(window_config.get("sparse_load", False)),
        )

    def process_input(
        self,
        data: RefinerModelInput,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        VarlenHandler,
        WindowLocalAttnHandler | None,
    ]:
        """Pack one refiner batch and construct its attention metadata.

        The denoising loop owns classifier-free guidance and diffusion times.
        ``UlyssesScheduler`` owns context-parallel sequence splitting.
        """
        batch_size, input_video_channel, video_time, video_height, video_width = data.x_t.shape
        video_tokens = self.img2tokens(data.x_t)
        reference_video_tokens = self.img2tokens(data.ref_video_feat)
        audio_tokens = data.audio_x_t.contiguous()
        reference_audio_tokens = data.ref_audio_feat.contiguous()
        text_tokens = data.txt_feat.contiguous()

        items: list[SingleData] = []
        for batch_index in range(batch_size):
            items.append(
                SingleData(
                    video_x_t=video_tokens[batch_index],
                    audio_x_t=audio_tokens[batch_index],
                    audio_feat_len=_to_int(data.audio_feat_len[batch_index]),
                    ref_audio_feat=reference_audio_tokens[batch_index],
                    ref_audio_feat_len=_to_int(data.ref_audio_feat_len[batch_index]),
                    txt_feat=text_tokens[batch_index],
                    txt_feat_len=_to_int(data.txt_feat_len[batch_index]),
                    t=video_time,
                    h=video_height,
                    w=video_width,
                    patch_size=self.patch_size,
                    t_patch_size=self.t_patch_size,
                    spatial_rope_interpolation=self.spatial_rope_interpolation,
                    text_offset=self.text_offset,
                    ref_video_feat=reference_video_tokens[batch_index],
                    ref_video_feat_len=_to_int(data.ref_video_feat_len[batch_index]),
                    coords_style=self.coords_style,
                )
            )
        packed_refiner_batch = PackedRefinerBatch(items=items)
        cumulative_lengths = packed_refiner_batch.cu_seqlen.to(
            device=data.x_t.device,
            dtype=torch.int32,
        )
        maximum_length = torch.tensor(
            packed_refiner_batch.max_seqlen,
            device=data.x_t.device,
            dtype=torch.int32,
        )
        varlen_handler = VarlenHandler(
            cu_seqlens_q=cumulative_lengths,
            cu_seqlens_k=cumulative_lengths.clone(),
            max_seqlen_q=maximum_length,
            max_seqlen_k=maximum_length.clone(),
        )

        token_sequence = packed_refiner_batch.token_sequence
        coords_mapping = packed_refiner_batch.coords_mapping
        modality_mapping = packed_refiner_batch.modality_mapping
        token_restore_order = None
        if self.use_window_attn:
            assert batch_size == 1, "window attention only supports batch size 1"
            item = packed_refiner_batch[0]
            local_attn_handler = self._build_window_local_attn_handler(
                item,
                int(packed_refiner_batch.total_token_num),
                token_sequence.device,
            )
            level = str(self.window_attn_config.get("level", "block")).lower()
            if level == "block":
                window_config = {**self.attn_config, **self.window_attn_config}
                block_sizes = (
                    int(window_config.get("block_t_size", self.attn_config.get("block_t_size", 1))),
                    int(window_config.get("block_size", self.attn_config.get("block_size", 1))),
                    int(window_config.get("block_size", self.attn_config.get("block_size", 1))),
                )
                patch_counts = (
                    item.t // item.t_patch_size,
                    item.h // item.patch_size,
                    item.w // item.patch_size,
                )
                token_order, token_restore_order, _, _ = self._build_block_order(
                    patch_counts=patch_counts,
                    block_sizes=block_sizes,
                    valid_token_num=int(packed_refiner_batch.total_token_num),
                    device=token_sequence.device,
                )
                token_sequence = token_sequence[token_order]
                coords_mapping = coords_mapping[token_order]
                modality_mapping = modality_mapping[token_order]
        elif self.frame_receptive_field != -1:
            assert batch_size == 1, "local attention only supports batch size 1"
            item = packed_refiner_batch[0]
            local_attn_handler = _build_frame_local_attn_handler(
                num_video_tokens=item.video_token_num,
                num_audio_and_text_tokens=(
                    item.audio_feat_len
                    + item.txt_feat_len
                    + item.ref_audio_feat_len
                ),
                num_frames=video_time,
                frame_receptive_field=self.frame_receptive_field,
                device=token_sequence.device,
            )
        else:
            local_attn_handler = None

        self.saved_for_output(
            packed_refiner_batch=packed_refiner_batch,
            input_video_channel=input_video_channel,
            token_restore_order=token_restore_order,
        )
        return (
            token_sequence,
            coords_mapping,
            modality_mapping,
            varlen_handler,
            local_attn_handler,
        )

    def process_output(
        self,
        output_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore block order and depack refiner video and audio outputs."""
        token_restore_order = self.get_saved_data("token_restore_order")
        if token_restore_order is not None:
            output_tokens = output_tokens[token_restore_order]
        packed_refiner_batch: PackedRefinerBatch = self.get_saved_data(
            "packed_refiner_batch"
        )
        return packed_refiner_batch.depack_token_sequence(output_tokens)
