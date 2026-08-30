# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""MiniMax H3 Video Sparse Attention for the native MLX runtime.

Ports the packed-sequence VSA-H3 contract from
``fastvideo/attention/backends/video_sparse_attn_h3.py``:

- Tiles are ``[segment-pure prefix chunks] + [3D video tiles]``.
- Tile sizes 64 ``(4, 4, 4)`` and 256 ``(4, 8, 8)``.
- Per-head pooled Q/K scoring and top-k routing.
- Prefix queries are always dense; prefix keys are ``exempt`` (always kept)
  or ``compete`` (FLOP-matched top-k).
- Optional dense-first steps and per-layer dense overrides.
- Trained ``to_gate_compress`` pooled-compression branch.

Execution backends:

- **reference** (``auto`` default) — grouped gather plus batched
  ``mx.fast.scaled_dot_product_attention``. Correctness baseline.
- **simd** — opt-in SIMD-group 8x8 matrix operations over the reference tile
  map for tile size 64 and head dimension 128. Unsupported shapes and kernel
  failures fall back to the reference backend.

Dense fused SDPA remains the default when VSA is disabled, the geometry is
unsupported, or a dense-only checkpoint is loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Integral
from typing import Any, Literal

import numpy as np

from fastvideo.logger import init_logger

logger = init_logger(__name__)

VSA_H3_TILE_SHAPES: dict[int, tuple[int, int, int]] = {
    256: (4, 8, 8),
    64: (4, 4, 4),
}
VSA_GATE_KEY_SUFFIX = "attn.to_gate_compress.weight"
VSA_PREFIX_MODES = ("exempt", "compete")
PUBLIC_VSA_IMPLS = ("auto", "reference", "simd")
VSA_IMPLS = PUBLIC_VSA_IMPLS
# SIMD is opt-in until its dynamically routed output is accepted as the default.
_AUTO_PREFERS_SIMD = False
_REFERENCE_FULL_MASK_TILE_LIMIT = 24
_REFERENCE_FULL_MASK_MAX_ELEMENTS = 256 * 1024**2

PrefixMode = Literal["exempt", "compete"]
VSAImpl = Literal["auto", "reference", "simd"]


class DenseOnlyVSACheckpointError(ValueError):
    """VSA was requested for a checkpoint that dropped the gate weights."""


@dataclass(frozen=True)
class MiniMaxH3VSAConfig:
    """Runtime VSA knobs. Defaults preserve dense MLX H3 behavior."""

    enabled: bool = False
    sparsity: float = 0.9
    tile_size: int = 64
    prefix_mode: PrefixMode = "exempt"
    dense_first_n_steps: int = 0
    dense_layers: tuple[int, ...] = ()
    impl: VSAImpl = "auto"

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"VSA sparsity must be in [0, 1), got {self.sparsity}.")
        if self.tile_size not in VSA_H3_TILE_SHAPES:
            raise ValueError(f"VSA tile_size must be one of {sorted(VSA_H3_TILE_SHAPES)}, got {self.tile_size}.")
        if self.prefix_mode not in VSA_PREFIX_MODES:
            raise ValueError(f"VSA prefix_mode must be one of {VSA_PREFIX_MODES}, got {self.prefix_mode!r}.")
        if self.impl not in VSA_IMPLS:
            raise ValueError(f"VSA impl must be one of {VSA_IMPLS}, got {self.impl!r}.")
        if self.dense_first_n_steps < 0:
            raise ValueError(f"vsa_dense_first_n_steps must be >= 0, got {self.dense_first_n_steps}.")
        if isinstance(self.dense_layers, str) or any(
                not isinstance(layer, Integral) or isinstance(layer, bool) or layer < 0 for layer in self.dense_layers):
            raise ValueError("VSA dense_layers must be a sequence of non-negative integer indices.")
        object.__setattr__(self, "dense_layers", tuple(int(layer) for layer in self.dense_layers))

    @property
    def exempt(self) -> bool:
        return self.prefix_mode == "exempt"

    def layer_sparsity(self, layer_idx: int, step_index: int) -> float:
        if not self.enabled:
            return 0.0
        if step_index < self.dense_first_n_steps:
            return 0.0
        if layer_idx in self.dense_layers:
            return 0.0
        return self.sparsity


@dataclass(frozen=True)
class MiniMaxH3VSAGeometry:
    """Packed-sequence tile map shared by routing, reference, and Metal paths."""

    prefix_segments: tuple[int, ...]
    dit_seq_shape: tuple[int, int, int]
    tile_shape: tuple[int, int, int]
    tile_elems: int
    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    variable_block_sizes: np.ndarray
    untile_combined_index: np.ndarray
    tile_partition_indices: np.ndarray

    @property
    def num_tiles(self) -> int:
        return int(self.variable_block_sizes.shape[0])

    @property
    def padded_length(self) -> int:
        return self.num_tiles * self.tile_elems

    @property
    def prefix_length(self) -> int:
        return int(sum(self.prefix_segments))

    @cached_property
    def tile_gather_index(self):
        import mlx.core as mx

        index = np.full(self.padded_length, self.total_seq_length, dtype=np.int32)
        index[self.untile_combined_index] = np.arange(self.total_seq_length, dtype=np.int32)
        return mx.array(index)

    @cached_property
    def prefix_gather_index(self):
        import mlx.core as mx

        index = np.full(self.num_prefix_tiles * self.tile_elems, self.prefix_length, dtype=np.int32)
        index[self.untile_combined_index[:self.prefix_length]] = np.arange(self.prefix_length, dtype=np.int32)
        return mx.array(index)

    @cached_property
    def untile_index(self):
        import mlx.core as mx

        return mx.array(self.untile_combined_index, dtype=mx.int32)


@dataclass
class MiniMaxH3VSAStats:
    """Filled during a sparse forward so the pipeline can report achieved sparsity."""

    configured_sparsity: float = 0.0
    layer_sparsity: float = 0.0
    tile_size: int = 64
    prefix_mode: str = "exempt"
    impl: str = "dense"
    num_prefix_tiles: int = 0
    num_video_tiles: int = 0
    video_keep: float = 0.0
    achieved_sparsity: float = 0.0
    dense_fallback_reason: str | None = None
    attention_calls: int = 0
    sparse_calls: int = 0
    impl_counts: dict[str, int] = field(default_factory=dict)
    fallback_reasons: list[str] = field(default_factory=list)

    def record(self, call: MiniMaxH3VSAStats) -> None:
        """Aggregate equally sized video-query tile maps across blocks and steps."""
        self.attention_calls += 1
        self.sparse_calls += int(call.layer_sparsity > 0.0)
        self.impl_counts[call.impl] = self.impl_counts.get(call.impl, 0) + 1
        self.impl = next(iter(self.impl_counts)) if len(self.impl_counts) == 1 else "mixed"
        for name in ("layer_sparsity", "video_keep", "achieved_sparsity"):
            old = getattr(self, name)
            setattr(self, name, old + (getattr(call, name) - old) / self.attention_calls)
        self.num_prefix_tiles = call.num_prefix_tiles
        self.num_video_tiles = call.num_video_tiles
        if call.dense_fallback_reason and call.dense_fallback_reason not in self.fallback_reasons:
            self.fallback_reasons.append(call.dense_fallback_reason)
        self.dense_fallback_reason = "; ".join(self.fallback_reasons) or None


def vsa_gate_key(block_index: int) -> str:
    return f"transformer_blocks.{block_index}.{VSA_GATE_KEY_SUFFIX}"


def expected_vsa_gate_keys(num_layers: int) -> tuple[str, ...]:
    return tuple(vsa_gate_key(index) for index in range(num_layers))


def is_vsa_gate_key(key: str) -> bool:
    return key.endswith(VSA_GATE_KEY_SUFFIX)


def dense_only_vsa_error(checkpoint_dir: str | Any) -> DenseOnlyVSACheckpointError:
    return DenseOnlyVSACheckpointError(
        f"VSA was requested but {checkpoint_dir} is a dense-only MLX H3 checkpoint "
        f"(no `{VSA_GATE_KEY_SUFFIX}` matrices / manifest vsa.capable). "
        "Reconvert with `--include-vsa`, for example:\n"
        "  python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \\\n"
        "    --model-root <FastH3 transformer dir> --out <new dir> --formats int6 --include-vsa")


def compute_topk(sparsity: float, num_blocks: int) -> int:
    """Blocks to keep for a sparsity level, clamped to [1, num_blocks]."""
    if num_blocks <= 0:
        return 0
    return max(1, min(math.ceil((1.0 - sparsity) * num_blocks), num_blocks))


def parse_dense_layers(value: str | None) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(layer < 0 for layer in layers):
        raise ValueError(f"vsa dense layers must be non-negative, got {value!r}.")
    return layers


def prefix_segments_from_layout(layout: Any, patch_size: tuple[int, int, int]) -> tuple[int, ...]:
    """Segment sizes preceding the generated-video tail, matching the PyTorch stage."""
    n_text = int(layout.text_indices.shape[0])
    n_cond = int(layout.num_condition_video_rows)
    n_audio = int(layout.audio_indices.shape[0])
    n_video = ((layout.num_video_latent_frames // patch_size[0]) * (layout.latent_height // patch_size[1]) *
               (layout.latent_width // patch_size[2]))
    if n_text + n_cond + n_audio + n_video != int(layout.sequence_length):
        raise ValueError("VSA-H3 supports the standard [text|cond|audio|video] packing only; "
                         f"segments ({n_text}, {n_cond}, {n_audio}) + video {n_video} do not sum to "
                         f"sequence length {layout.sequence_length}.")
    return n_text, n_cond, n_audio


def dit_seq_shape_from_layout(layout: Any, patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        int(layout.num_video_latent_frames // patch_size[0]),
        int(layout.latent_height // patch_size[1]),
        int(layout.latent_width // patch_size[2]),
    )


def _video_tile_sizes(dit_seq_shape: tuple[int, int, int], tile_shape: tuple[int, int, int]) -> np.ndarray:
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = tile_shape
    n_t, n_h, n_w = math.ceil(t / ts_t), math.ceil(h / ts_h), math.ceil(w / ts_w)

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> np.ndarray:
        sizes = np.full((n_tiles, ), tile, dtype=np.int64)
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)
    h_sizes = _sizes(h, ts_h, n_h)
    w_sizes = _sizes(w, ts_w, n_w)
    return (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).reshape(-1)


def _video_tile_partition_indices(dit_seq_shape: tuple[int, int, int], tile_shape: tuple[int, int, int]) -> np.ndarray:
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = tile_shape
    indices = np.arange(t * h * w, dtype=np.int64).reshape(t, h, w)
    chunks: list[np.ndarray] = []
    for tt in range(math.ceil(t / ts_t)):
        for hh in range(math.ceil(h / ts_h)):
            for ww in range(math.ceil(w / ts_w)):
                chunks.append(indices[tt * ts_t:min(tt * ts_t + ts_t, t), hh * ts_h:min(hh * ts_h + ts_h, h),
                                      ww * ts_w:min(ww * ts_w + ts_w, w)].reshape(-1))
    return np.concatenate(chunks, axis=0)


def _non_pad_index(variable_block_sizes: np.ndarray, tile_elems: int) -> np.ndarray:
    n_win = int(variable_block_sizes.shape[0])
    starts = np.arange(n_win, dtype=np.int64) * tile_elems
    index_pad = starts[:, None] + np.arange(tile_elems, dtype=np.int64)[None, :]
    index_mask = np.arange(tile_elems, dtype=np.int64)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


def validate_h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    variable_block_sizes: np.ndarray,
    untile_combined_index: np.ndarray,
    tile_elems: int,
) -> None:
    total = sum(prefix_segments) + math.prod(dit_seq_shape)
    n_pad = int(variable_block_sizes.size) * tile_elems
    sizes_min = int(variable_block_sizes.min()) if variable_block_sizes.size else 0
    sizes_max = int(variable_block_sizes.max()) if variable_block_sizes.size else 0
    sizes_sum = int(variable_block_sizes.sum())
    if sizes_min < 1 or sizes_max > tile_elems or sizes_sum != total:
        raise ValueError(f"VSA-H3 tile sizes out of bounds for prefix={prefix_segments}, video={dit_seq_shape}, "
                         f"tile_elems={tile_elems}: min={sizes_min}, max={sizes_max}, sum={sizes_sum}, "
                         f"expected sum={total}.")
    if int(untile_combined_index.size) != total:
        raise ValueError(f"VSA-H3 untile index has {untile_combined_index.size} entries for a packed "
                         f"sequence of {total} rows (prefix={prefix_segments}, video={dit_seq_shape}).")
    idx_min = int(untile_combined_index.min())
    idx_max = int(untile_combined_index.max())
    if idx_min < 0 or idx_max >= n_pad:
        raise ValueError(f"VSA-H3 untile index is not an injective map into non-pad slots: range "
                         f"[{idx_min}, {idx_max}] vs padded length {n_pad} "
                         f"(prefix={prefix_segments}, video={dit_seq_shape}).")
    in_tile_offset = untile_combined_index % tile_elems
    maps_into_pad = bool((in_tile_offset >= variable_block_sizes[untile_combined_index // tile_elems]).any())
    if maps_into_pad or int(np.unique(untile_combined_index).size) != total:
        raise ValueError(f"VSA-H3 untile index is not an injective map into non-pad slots: "
                         f"pad-slot hit={maps_into_pad} "
                         f"(prefix={prefix_segments}, video={dit_seq_shape}).")


def build_h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    tile_size: int = 64,
) -> MiniMaxH3VSAGeometry:
    """Tile the packed sequence: segment-pure prefix chunks, then video tiles."""
    tile_shape = VSA_H3_TILE_SHAPES.get(int(tile_size))
    if tile_shape is None:
        raise ValueError(f"VSA-H3 tile_size must be one of {sorted(VSA_H3_TILE_SHAPES)}, got {tile_size!r}")
    if len(dit_seq_shape) != 3 or any(axis <= 0 for axis in dit_seq_shape):
        raise ValueError(f"VSA-H3 video axes must be positive, got {dit_seq_shape}.")
    tile_elems = math.prod(tile_shape)
    prefix_segments = tuple(int(segment) for segment in prefix_segments if int(segment) > 0)
    prefix_len = sum(prefix_segments)

    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, rem = divmod(segment, tile_elems)
        prefix_sizes.extend([tile_elems] * full)
        if rem:
            prefix_sizes.append(rem)
    num_prefix_tiles = len(prefix_sizes)

    video_sizes = _video_tile_sizes(dit_seq_shape, tile_shape)
    num_video_tiles = int(video_sizes.size)
    video_indices = _video_tile_partition_indices(dit_seq_shape, tile_shape) + prefix_len
    tile_partition_indices = np.concatenate(
        [np.arange(prefix_len, dtype=np.int64), video_indices],
        axis=0,
    )
    variable_block_sizes = np.concatenate(
        [np.asarray(prefix_sizes, dtype=np.int64),
         video_sizes.astype(np.int64)],
        axis=0,
    )
    non_pad_index = _non_pad_index(variable_block_sizes, tile_elems)
    untile_combined_index = non_pad_index[np.argsort(tile_partition_indices, kind="stable")]
    validate_h3_tile_geometry(prefix_segments, dit_seq_shape, variable_block_sizes, untile_combined_index, tile_elems)
    return MiniMaxH3VSAGeometry(
        prefix_segments=prefix_segments,
        dit_seq_shape=dit_seq_shape,
        tile_shape=tile_shape,
        tile_elems=tile_elems,
        total_seq_length=prefix_len + math.prod(dit_seq_shape),
        num_prefix_tiles=num_prefix_tiles,
        num_video_tiles=num_video_tiles,
        variable_block_sizes=variable_block_sizes,
        untile_combined_index=untile_combined_index,
        tile_partition_indices=tile_partition_indices,
    )


def token_tile_and_valid(variable_block_sizes: np.ndarray, tile_elems: int) -> tuple[np.ndarray, np.ndarray]:
    token_tile = np.repeat(np.arange(variable_block_sizes.size, dtype=np.int64), tile_elems)
    token_valid = (np.arange(tile_elems, dtype=np.int64)[None, :] < variable_block_sizes[:, None]).reshape(-1)
    return token_tile, token_valid


def build_block_mask(
    scores: np.ndarray,
    num_prefix_tiles: int,
    num_video_tiles: int,
    sparsity: float,
    exempt: bool,
) -> np.ndarray:
    """scores: [..., n_tiles, n_tiles] -> bool mask, same shape.

    Mirrors ``_build_block_mask`` in the PyTorch H3 backend.
    """
    n_tiles = scores.shape[-1]
    k_vid = compute_topk(sparsity, num_video_tiles)
    if k_vid == num_video_tiles:
        return np.ones_like(scores, dtype=bool)
    mask = np.zeros_like(scores, dtype=bool)
    if exempt or num_prefix_tiles == 0:
        video_cols = scores[..., num_prefix_tiles:]
        idx = np.argsort(-video_cols, axis=-1)[..., :k_vid] + num_prefix_tiles
        np.put_along_axis(mask, idx, True, axis=-1)
        mask[..., :num_prefix_tiles] = True
    else:
        k_total = min(k_vid + num_prefix_tiles, n_tiles)
        idx = np.argsort(-scores, axis=-1)[..., :k_total]
        np.put_along_axis(mask, idx, True, axis=-1)
    mask[..., :num_prefix_tiles, :] = True
    return mask


def _tile_hidden(x, geometry: MiniMaxH3VSAGeometry):
    """Gather packed rows into tiles, mapping every padding slot to one zero row."""
    import mlx.core as mx

    seq, heads, dim = x.shape
    if seq != geometry.total_seq_length:
        raise ValueError(f"VSA-H3 metadata was built for sequence length {geometry.total_seq_length}, got {seq}.")
    return mx.concatenate([x, mx.zeros((1, heads, dim), dtype=x.dtype)])[geometry.tile_gather_index]


def _untile_hidden(tiled, geometry: MiniMaxH3VSAGeometry):
    return tiled[geometry.untile_index]


def _pool_tiles(x, variable_block_sizes, tile_elems: int):
    """fp32 mean over each tile. x: [S_pad, H, D] -> [H, n_tiles, D]."""
    import mlx.core as mx

    seq_len, heads, dim = x.shape
    n_tiles = seq_len // tile_elems
    pooled = x.astype(mx.float32).reshape(n_tiles, tile_elems, heads, dim).sum(axis=1)
    pooled = pooled / mx.array(variable_block_sizes, dtype=mx.float32)[:, None, None]
    return pooled.transpose(1, 0, 2)


def _dense_sdpa(q, k, v, scale: float):
    import mlx.core as mx

    return mx.fast.scaled_dot_product_attention(
        q.transpose(1, 0, 2)[None],
        k.transpose(1, 0, 2)[None],
        v.transpose(1, 0, 2)[None],
        scale=scale,
    )[0].transpose(1, 0, 2)


def _selected_keep(sparsity: float, num_prefix_tiles: int, num_video_tiles: int, exempt: bool) -> int:
    k_vid = compute_topk(sparsity, num_video_tiles)
    if k_vid == num_video_tiles:
        return num_prefix_tiles + num_video_tiles
    if exempt or num_prefix_tiles == 0:
        return num_prefix_tiles + k_vid
    return min(k_vid + num_prefix_tiles, num_prefix_tiles + num_video_tiles)


def _block_indices_from_scores(
    scores,
    num_prefix_tiles: int,
    num_video_tiles: int,
    sparsity: float,
    exempt: bool,
):
    """Return (block_idx [H, n_video_tiles, k_sel], block_num [H, n_video_tiles])."""
    import mlx.core as mx

    n_tiles = scores.shape[-1]
    heads = scores.shape[0]
    k_vid = compute_topk(sparsity, num_video_tiles)
    video_scores = scores[:, num_prefix_tiles:, :]
    if k_vid == num_video_tiles:
        idx = mx.broadcast_to(mx.arange(n_tiles, dtype=mx.int32)[None, None, :], (heads, num_video_tiles, n_tiles))
        counts = mx.full((heads, num_video_tiles), n_tiles, dtype=mx.int32)
        return idx, counts

    if exempt or num_prefix_tiles == 0:
        k_sel = num_prefix_tiles + k_vid
        video_cols = video_scores[:, :, num_prefix_tiles:]
        top = mx.argpartition(-video_cols, kth=k_vid - 1, axis=-1)[:, :, :k_vid].astype(mx.int32) + num_prefix_tiles
        if num_prefix_tiles:
            prefix = mx.broadcast_to(
                mx.arange(num_prefix_tiles, dtype=mx.int32)[None, None, :],
                (heads, num_video_tiles, num_prefix_tiles),
            )
            idx = mx.concatenate([prefix, top], axis=-1)
        else:
            idx = top
    else:
        k_sel = min(k_vid + num_prefix_tiles, n_tiles)
        top = mx.argpartition(-video_scores, kth=k_sel - 1, axis=-1)[:, :, :k_sel].astype(mx.int32)
        idx = top
    counts = mx.full((heads, num_video_tiles), idx.shape[-1], dtype=mx.int32)
    return idx, counts


def _token_mask_from_block_mask(block_mask: np.ndarray, geometry: MiniMaxH3VSAGeometry) -> np.ndarray:
    token_tile, token_valid = token_tile_and_valid(geometry.variable_block_sizes, geometry.tile_elems)
    # block_mask: [H, n_tiles, n_tiles]
    allow = block_mask[:, token_tile][:, :, token_tile] & token_valid[None, None, :]
    return allow


def _reference_token_sdpa(q_tiled, k_tiled, v_tiled, block_mask: np.ndarray, geometry: MiniMaxH3VSAGeometry,
                          scale: float):
    import mlx.core as mx

    allow = _token_mask_from_block_mask(block_mask, geometry)
    return mx.fast.scaled_dot_product_attention(
        mx.contiguous(q_tiled.transpose(1, 0, 2))[None],
        mx.contiguous(k_tiled.transpose(1, 0, 2))[None],
        mx.contiguous(v_tiled.transpose(1, 0, 2))[None],
        scale=scale,
        mask=mx.array(allow)[None],
    )[0].transpose(1, 0, 2)


def _reference_full_mask_fits(geometry: MiniMaxH3VSAGeometry, heads: int) -> bool:
    """Bound the dense token mask by both tile count and materialized size."""
    mask_elements = heads * geometry.padded_length**2
    return (geometry.num_tiles <= _REFERENCE_FULL_MASK_TILE_LIMIT
            and mask_elements <= _REFERENCE_FULL_MASK_MAX_ELEMENTS)


def _gather_selected_kv(k_tiles, v_tiles, block_idx, tile_elems: int):
    """Gather K/V tiles for one query-tile batch.

    k_tiles/v_tiles: [H, n_tiles, tile_elems, D]
    block_idx: [H, n_q, k_sel]
    returns K/V: [H, n_q, k_sel * tile_elems, D]
    """
    import mlx.core as mx

    heads, n_q, k_sel = block_idx.shape
    dim = k_tiles.shape[-1]
    gathered_k = k_tiles[mx.arange(heads)[:, None, None], block_idx]
    gathered_v = v_tiles[mx.arange(heads)[:, None, None], block_idx]
    return (
        gathered_k.reshape(heads, n_q, k_sel * tile_elems, dim),
        gathered_v.reshape(heads, n_q, k_sel * tile_elems, dim),
    )


def _key_valid_mask(block_idx, variable_block_sizes, tile_elems: int):
    import mlx.core as mx

    vbs = mx.array(variable_block_sizes, dtype=mx.int32)
    selected_sizes = vbs[block_idx]  # [H, n_q, k_sel]
    offsets = mx.arange(tile_elems, dtype=mx.int32)
    return offsets[None, None, None, :] < selected_sizes[:, :, :, None]


_REFERENCE_GATHER_TARGET_BYTES = 2 * 1024**3


def _reference_gather_query_chunk(heads: int, dim: int, k_sel: int, tile_elems: int, n_q: int) -> int:
    """Batch as many query tiles as fit in ~2 GiB of gathered BF16 K/V."""
    bytes_per_query = 4 * heads * max(k_sel, 1) * tile_elems * dim
    chunk = min(n_q, max(1, _REFERENCE_GATHER_TARGET_BYTES // max(bytes_per_query, 1)))
    return int(chunk)


def _reference_gather_sdpa(
    q_tiled,
    k_tiled,
    v_tiled,
    block_idx,
    geometry: MiniMaxH3VSAGeometry,
    scale: float,
):
    """Grouped gather + batched SDPA over video query tiles.

    Full-sequence gather at 720p materializes tens of GiB of selected K/V, so
    query tiles are processed in memory-bounded chunks. This is the correctness
    baseline and the default ``auto`` path.
    """
    import mlx.core as mx

    tile_elems = geometry.tile_elems
    heads, dim = q_tiled.shape[1], q_tiled.shape[2]
    n_tiles = geometry.num_tiles
    q_tiles = q_tiled.reshape(n_tiles, tile_elems, heads, dim).transpose(2, 0, 1, 3)
    k_tiles = k_tiled.reshape(n_tiles, tile_elems, heads, dim).transpose(2, 0, 1, 3)
    v_tiles = v_tiled.reshape(n_tiles, tile_elems, heads, dim).transpose(2, 0, 1, 3)
    q_video = q_tiles[:, geometry.num_prefix_tiles:]  # [H, V, tile_elems, D]
    n_q = int(q_video.shape[1])
    k_sel = int(block_idx.shape[-1])
    query_chunk = _reference_gather_query_chunk(heads, dim, k_sel, tile_elems, n_q)
    chunks = []
    for start in range(0, n_q, query_chunk):
        end = min(start + query_chunk, n_q)
        idx = block_idx[:, start:end, :]
        n_chunk = end - start
        k_sel = int(idx.shape[-1])
        gathered_k, gathered_v = _gather_selected_kv(k_tiles, v_tiles, idx, tile_elems)
        valid = _key_valid_mask(idx, geometry.variable_block_sizes, tile_elems)
        valid = valid.reshape(heads, n_chunk, k_sel * tile_elems)
        q_bh = mx.contiguous(q_video[:, start:end].reshape(heads * n_chunk, tile_elems, dim)[None])
        k_bh = mx.contiguous(gathered_k.reshape(heads * n_chunk, k_sel * tile_elems, dim)[None])
        v_bh = mx.contiguous(gathered_v.reshape(heads * n_chunk, k_sel * tile_elems, dim)[None])
        mask = valid.reshape(1, heads * n_chunk, 1, k_sel * tile_elems)
        out = mx.fast.scaled_dot_product_attention(q_bh, k_bh, v_bh, scale=scale, mask=mask)[0]
        out = out.reshape(heads, n_chunk, tile_elems, dim)
        mx.eval(out)
        chunks.append(out)
    out = mx.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
    return out.transpose(1, 2, 0, 3).reshape(n_q * tile_elems, heads, dim)


def _gate_compress_output(scores, v_tiled, gate_tiled, geometry: MiniMaxH3VSAGeometry):
    import mlx.core as mx

    v_pooled = _pool_tiles(v_tiled, geometry.variable_block_sizes, geometry.tile_elems)
    attn = mx.softmax(scores, axis=-1)
    out_c = attn @ v_pooled  # [H, n_tiles, D]
    out_c = out_c.transpose(1, 0, 2)  # [n_tiles, H, D]
    heads, dim = gate_tiled.shape[1], gate_tiled.shape[2]
    gate_view = gate_tiled.reshape(geometry.num_tiles, geometry.tile_elems, heads, dim)
    return (out_c[:, None, :, :] * gate_view).reshape(geometry.padded_length, heads, dim)


def resolve_impl(requested: VSAImpl, head_dim: int, tile_elems: int) -> str:
    if requested == "reference" or (requested == "auto" and not _AUTO_PREFERS_SIMD):
        return "reference"

    from fastvideo.mlx_runtime.minimax_h3_vsa_simd import simd_kernel_available, simd_kernel_error

    if tile_elems != 64 or head_dim != 128:
        if requested == "simd":
            logger.warning_once(
                f"SIMD VSA requires tile 64 and head dim 128 (got tile={tile_elems} head_dim={head_dim}); "
                "using reference VSA")
        return "reference"
    if not simd_kernel_available():
        reason = simd_kernel_error() or "SIMD-group VSA kernel is unavailable"
        logger.warning_once(f"SIMD VSA kernel unavailable ({reason}); using reference VSA")
        return "reference"
    return "simd"


def h3_vsa_attention(
    query,
    key,
    value,
    geometry: MiniMaxH3VSAGeometry,
    *,
    sparsity: float,
    exempt: bool = True,
    gate_compress=None,
    impl: VSAImpl = "auto",
    stats: MiniMaxH3VSAStats | None = None,
):
    """Packed ``[S, H, D]`` VSA attention. Falls back to dense SDPA when sparsity is 0."""
    import mlx.core as mx

    _, heads, dim = query.shape
    scale = dim**-0.5
    if stats is not None:
        stats.configured_sparsity = sparsity
        stats.layer_sparsity = sparsity
        stats.tile_size = geometry.tile_elems
        stats.prefix_mode = "exempt" if exempt else "compete"
        stats.num_prefix_tiles = geometry.num_prefix_tiles
        stats.num_video_tiles = geometry.num_video_tiles

    if sparsity <= 0.0 and gate_compress is None:
        if stats is not None:
            stats.impl = "dense"
            stats.achieved_sparsity = 0.0
            stats.video_keep = geometry.num_video_tiles
        return _dense_sdpa(query, key, value, scale)

    q_tiled = _tile_hidden(query, geometry)
    k_tiled = _tile_hidden(key, geometry)
    v_tiled = _tile_hidden(value, geometry)
    q_pooled = _pool_tiles(q_tiled, geometry.variable_block_sizes, geometry.tile_elems)
    k_pooled = _pool_tiles(k_tiled, geometry.variable_block_sizes, geometry.tile_elems)
    scores = (q_pooled @ k_pooled.transpose(0, 2, 1)) / (dim**0.5)

    k_vid = compute_topk(sparsity, geometry.num_video_tiles)
    if stats is not None:
        stats.video_keep = k_vid if sparsity > 0.0 else geometry.num_video_tiles
        stats.achieved_sparsity = (1.0 - (k_vid / geometry.num_video_tiles) if geometry.num_video_tiles else 0.0)

    if sparsity <= 0.0:
        out_tiled = _tile_hidden(_dense_sdpa(query, key, value, scale), geometry)
        chosen = "dense"
    else:
        block_idx, block_num = _block_indices_from_scores(
            scores,
            geometry.num_prefix_tiles,
            geometry.num_video_tiles,
            sparsity,
            exempt,
        )
        if stats is not None and not exempt:
            stats.video_keep = float(mx.mean(mx.sum(block_idx >= geometry.num_prefix_tiles, axis=-1)).item())
            stats.achieved_sparsity = 1.0 - stats.video_keep / geometry.num_video_tiles
        chosen = resolve_impl(impl, dim, geometry.tile_elems)
        if stats is not None and impl == "simd" and chosen != "simd":
            from fastvideo.mlx_runtime.minimax_h3_vsa_simd import simd_kernel_error

            stats.dense_fallback_reason = ("SIMD requires tile 64 and head dim 128"
                                           if geometry.tile_elems != 64 or dim != 128 else simd_kernel_error())
        prefix_out = (_dense_sdpa(query[:geometry.prefix_length], key, value, scale)
                      if geometry.prefix_length else query[:0])
        n_prefix_pad = geometry.num_prefix_tiles * geometry.tile_elems
        if chosen == "simd":
            from fastvideo.mlx_runtime.minimax_h3_vsa_simd import disable_simd_kernel, simd_block_sparse

            try:
                video_tiled = simd_block_sparse(q_tiled, k_tiled, v_tiled, block_idx, block_num, geometry, scale)
                # MLX compiles and executes custom Metal kernels lazily.
                mx.eval(video_tiled)
                video_tiled = video_tiled[n_prefix_pad:]
            except Exception as error:  # noqa: BLE001 - keep generation alive on kernel failure
                disable_simd_kernel(error)
                logger.warning_once(f"SIMD VSA kernel failed ({error}); falling back to reference gather+SDPA")
                chosen = "reference"
                if stats is not None:
                    stats.dense_fallback_reason = f"simd kernel failed: {error}"
                video_tiled = _reference_gather_sdpa(q_tiled, k_tiled, v_tiled, block_idx, geometry, scale)
        elif _reference_full_mask_fits(geometry, heads):
            scores_np = np.array(scores, dtype=np.float32)
            mask = build_block_mask(
                scores_np,
                geometry.num_prefix_tiles,
                geometry.num_video_tiles,
                sparsity,
                exempt,
            )
            video_tiled = _reference_token_sdpa(q_tiled, k_tiled, v_tiled, mask, geometry, scale)[n_prefix_pad:]
        else:
            video_tiled = _reference_gather_sdpa(q_tiled, k_tiled, v_tiled, block_idx, geometry, scale)
        video_tiled = video_tiled.astype(query.dtype)
        prefix_tiled = mx.concatenate([
            prefix_out,
            mx.zeros((1, heads, dim), dtype=query.dtype),
        ])[geometry.prefix_gather_index]
        # Prefix query tiles are dense; keep fused-SDPA prefix rows and sparse video tiles.
        out_tiled = mx.concatenate([prefix_tiled, video_tiled], axis=0)

    if gate_compress is not None:
        gate_tiled = _tile_hidden(gate_compress, geometry)
        out_tiled = out_tiled + _gate_compress_output(scores, v_tiled, gate_tiled, geometry).astype(out_tiled.dtype)

    if stats is not None:
        stats.impl = chosen if sparsity > 0.0 else "dense"
    return _untile_hidden(out_tiled, geometry)


def geometry_is_supported(prefix_segments: tuple[int, ...], dit_seq_shape: tuple[int, int, int],
                          tile_size: int) -> str | None:
    """Return a fallback reason, or None when VSA can run."""
    try:
        build_h3_tile_geometry(prefix_segments, dit_seq_shape, tile_size)
    except ValueError as error:
        return str(error)
    return None
