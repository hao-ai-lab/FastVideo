# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""Prefix K/V reuse across denoise steps for MiniMax H3 Ref2VA (native MLX).

A Ref2VA packed sequence is ``[text | ordered references | target audio |
target video]``. The text and reference rows are conditioning: their AdaLN
timestep and their input embeddings never change between denoise steps. They
do attend (bidirectionally) to the noisy target rows, so their per-layer
hidden states drift slightly; reuse freezes them at the step that recorded the
cache. That is an approximation, hence opt-in.

``MiniMaxH3PrefixKVCache`` holds each block's post-norm, post-rotary K and V
for the frozen rows. Steps that reuse it run every block on the target rows
only (see ``minimax_h3_prefix_kv_forward``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PREFIX_KV_STORAGE = ("bf16", "fp16", "int8", "native")
DEFAULT_PREFIX_KV_MAX_GIB = 4.0
_GIB = 1024**3
_INT8_MAX = 127.0


class PrefixKVBudgetError(ValueError):
    """The estimated prefix K/V cache exceeds the configured memory budget."""


def validate_prefix_kv_storage(storage: str) -> str:
    if storage not in PREFIX_KV_STORAGE:
        raise ValueError(f"prefix K/V storage must be one of {PREFIX_KV_STORAGE}, got {storage!r}.")
    return storage


def estimate_prefix_kv_bytes(
    num_layers: int,
    rows: int,
    num_heads: int,
    head_dim: int,
    storage: str = "bf16",
    native_itemsize: int = 2,
) -> int:
    """``2 (K and V) * layers * rows * heads * head_dim * itemsize`` (+ int8 scales)."""
    validate_prefix_kv_storage(storage)
    if storage == "int8":
        per_row = num_heads * head_dim + 2 * num_heads  # int8 payload + fp16 scale per (row, head)
    else:
        per_row = num_heads * head_dim * (native_itemsize if storage == "native" else 2)
    return 2 * num_layers * rows * per_row


def check_prefix_kv_budget(estimated_bytes: int, max_gib: float, *, rows: int) -> None:
    if max_gib <= 0 or not math.isfinite(max_gib):
        raise ValueError(f"prefix_kv_cache_max_gib must be positive, got {max_gib}.")
    if estimated_bytes > max_gib * _GIB:
        raise PrefixKVBudgetError(
            f"The prefix K/V cache for {rows} frozen rows needs ~{estimated_bytes / _GIB:.2f} GiB, over the "
            f"prefix_kv_cache_max_gib budget of {max_gib:g} GiB. Raise the budget, shrink the references "
            "(for example reference_max_pixels='match'), use int8 storage, or run without prefix K/V reuse.")


def frozen_prefix_rows(layout: Any) -> int:
    """Rows before the target audio, after checking the tail is ``[target audio | target video]``."""
    if not getattr(layout, "reference_segments", ()):
        raise ValueError("Prefix K/V reuse needs a Ref2VA packed layout (it carries per-reference spans).")
    num_audio = int(layout.audio_indices.shape[0]) - int(layout.num_condition_audio_rows)
    num_video = int(layout.video_indices.shape[0]) - int(layout.num_condition_video_rows)
    frozen = int(layout.sequence_length) - num_audio - num_video
    audio_tail = np.arange(frozen, frozen + num_audio)
    video_tail = np.arange(frozen + num_audio, int(layout.sequence_length))
    if (frozen <= 0 or num_audio <= 0 or num_video <= 0
            or not np.array_equal(layout.audio_indices[int(layout.num_condition_audio_rows):], audio_tail)
            or not np.array_equal(layout.video_indices[int(layout.num_condition_video_rows):], video_tail)):
        raise ValueError("Prefix K/V reuse needs the target audio and video rows to form the packed tail.")
    return frozen


@dataclass
class MiniMaxH3PrefixKVCache:
    """Per-block K/V of the frozen rows ([rows, heads, head_dim]) in compact storage."""

    num_layers: int
    rows: int
    num_heads: int
    head_dim: int
    storage: str = "bf16"
    compute_dtype: Any = None  # dtype of the recorded K/V (what blocks consume)
    hidden_dtype: Any = None  # dtype of the packed hidden states at record time
    recorded_step: int = -1
    keys: list[Any] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_prefix_kv_storage(self.storage)
        self.reset()

    def reset(self) -> None:
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers

    @property
    def complete(self) -> bool:
        return all(entry is not None for entry in self.keys) and all(entry is not None for entry in self.values)

    @property
    def nbytes(self) -> int:
        total = 0
        for entry in (*self.keys, *self.values):
            if entry is not None:
                total += sum(int(part.nbytes) for part in entry if part is not None)
        return total

    def _encode(self, x) -> tuple[Any, Any]:
        import mlx.core as mx

        if self.storage == "int8":
            absmax = mx.max(mx.abs(x.astype(mx.float32)), axis=-1, keepdims=True)
            scale = mx.maximum(absmax / _INT8_MAX, 1e-8).astype(mx.float16)
            quantized = mx.clip(mx.round(x.astype(mx.float32) / scale.astype(mx.float32)), -_INT8_MAX,
                                _INT8_MAX).astype(mx.int8)
            return quantized, scale
        # ``x * 1`` forces a fresh buffer: a bare row slice (or a same-dtype astype) would be a view that keeps
        # the whole [S, H, D] K/V of the block alive for the entire denoise.
        if self.storage == "native":
            return x * 1, None
        return (x * 1).astype(mx.bfloat16 if self.storage == "bf16" else mx.float16), None

    def _decode(self, entry: tuple[Any, Any]):
        data, scale = entry
        if scale is not None:
            return (data.astype(scale.dtype) * scale).astype(self.compute_dtype)
        return data.astype(self.compute_dtype)

    def store(self, layer: int, key, value) -> list[Any]:
        """Keep the frozen rows of one block's K/V; returns arrays to evaluate."""
        if key.shape[0] != self.rows or value.shape[0] != self.rows:
            raise ValueError(f"Expected {self.rows} frozen K/V rows, got {key.shape[0]}/{value.shape[0]}.")
        self.compute_dtype = key.dtype
        self.keys[layer] = self._encode(key)
        self.values[layer] = self._encode(value)
        return [part for entry in (self.keys[layer], self.values[layer]) for part in entry if part is not None]

    def load(self, layer: int) -> tuple[Any, Any]:
        if self.keys[layer] is None or self.values[layer] is None:
            raise RuntimeError(f"Prefix K/V for block {layer} was never recorded.")
        return self._decode(self.keys[layer]), self._decode(self.values[layer])


@dataclass(frozen=True)
class TargetQueryPlan:
    """Tile bookkeeping to run VSA for the target rows only.

    The target rows are the packed tail ``[target audio | target video]``.
    Their queries live in the last dense (audio) tiles and the last sparse
    region (the generated video); ``query_tiles`` lists those tile ids in that
    order, ``tile_gather`` maps each query-tile slot to a target-row index
    (``num_target_rows`` marks padding) and ``out_slot`` is its inverse.
    """

    frozen_rows: int
    num_audio_rows: int
    num_video_rows: int
    num_audio_tiles: int
    query_tiles: np.ndarray
    tile_gather: Any
    out_slot: Any
    audio_tile_gather: Any

    @property
    def num_target_rows(self) -> int:
        return self.num_audio_rows + self.num_video_rows


def build_target_query_plan(geometry: Any, frozen_rows: int, num_audio_rows: int,
                            num_video_rows: int) -> TargetQueryPlan:
    import mlx.core as mx

    cached = geometry.__dict__.setdefault("_target_query_plans", {})
    key = (frozen_rows, num_audio_rows, num_video_rows)
    if key in cached:
        return cached[key]
    tile = geometry.tile_elems
    total = geometry.total_seq_length
    if frozen_rows + num_audio_rows + num_video_rows != total:
        raise ValueError(f"Target rows {num_audio_rows}+{num_video_rows} after {frozen_rows} frozen rows do not "
                         f"cover the VSA geometry's {total} rows.")
    num_audio_tiles = math.ceil(num_audio_rows / tile) if num_audio_rows else 0
    video_start, video_end = geometry.video_tile_spans[-1]
    audio_tiles = np.arange(geometry.num_prefix_tiles - num_audio_tiles, geometry.num_prefix_tiles, dtype=np.int64)
    query_tiles = np.concatenate([audio_tiles, np.arange(video_start, video_end, dtype=np.int64)])
    if int(geometry.variable_block_sizes[query_tiles].sum()) != num_audio_rows + num_video_rows:
        raise ValueError("VSA geometry's trailing tiles do not hold exactly the target audio and video rows.")

    slot_row = np.full(geometry.padded_length, total, dtype=np.int64)
    slot_row[geometry.untile_combined_index] = np.arange(total, dtype=np.int64)
    rows = slot_row.reshape(geometry.num_tiles, tile)[query_tiles].reshape(-1)
    target_rows = np.where((rows >= frozen_rows) & (rows < total), rows - frozen_rows, num_audio_rows + num_video_rows)
    lookup = np.full(geometry.num_tiles, -1, dtype=np.int64)
    lookup[query_tiles] = np.arange(query_tiles.size)
    packed_slots = geometry.untile_combined_index[frozen_rows:]
    out_slot = lookup[packed_slots // tile] * tile + packed_slots % tile
    audio_rows = target_rows[:num_audio_tiles * tile]
    audio_rows = np.where(audio_rows < num_audio_rows, audio_rows, num_audio_rows)
    plan = TargetQueryPlan(
        frozen_rows=frozen_rows,
        num_audio_rows=num_audio_rows,
        num_video_rows=num_video_rows,
        num_audio_tiles=num_audio_tiles,
        query_tiles=query_tiles,
        tile_gather=mx.array(target_rows.astype(np.int32)),
        out_slot=mx.array(out_slot.astype(np.int32)),
        audio_tile_gather=mx.array(audio_rows.astype(np.int32)),
    )
    cached[key] = plan
    return plan
