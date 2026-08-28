# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""Inference-only MiniMax H3 Video Sparse Attention for the MLX runtime.

Ports the packed-sequence VSA-H3 contract from
``fastvideo/attention/backends/video_sparse_attn_h3.py``:

- Tiles are ``[segment-pure prefix chunks] + [3D video tiles]``.
- Tile sizes 64 ``(4, 4, 4)`` and 256 ``(4, 8, 8)``.
- Per-head pooled Q/K scoring and top-k routing.
- Prefix queries are always dense; prefix keys are ``exempt`` (always kept)
  or ``compete`` (FLOP-matched top-k).
- Optional dense-first steps and per-layer dense overrides.
- Trained ``to_gate_compress`` pooled-compression branch.

Two execution paths:

- **reference** (``auto`` default) — grouped gather plus batched
  ``mx.fast.scaled_dot_product_attention``. Correctness baseline. On MLX 0.31.2
  / M4 Max this is close to dense fused SDPA at 720p; the Metal kernel is not.
- **metal** — inference-only ``mx.fast.metal_kernel`` block-sparse attention
  over BF16 Q/K/V activations and an explicit per-query block index. Kept for
  parity testing. INT6 applies only to linear weights (including the gate
  projection), not to Q/K/V. ``compile_options`` is used on MLX 0.32.2+ and
  omitted on 0.31.2.

Dense fused SDPA remains the default when VSA is disabled, the geometry is
unsupported, or a dense-only checkpoint is loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
VSA_IMPLS = ("auto", "reference", "metal")
_METAL_MAX_HEAD_DIM = 128
_REFERENCE_FULL_MASK_TILE_LIMIT = 24

PrefixMode = Literal["exempt", "compete"]
VSAImpl = Literal["auto", "reference", "metal"]


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
    video_keep: int = 0
    achieved_sparsity: float = 0.0
    dense_fallback_reason: str | None = None


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
    """Scatter packed rows ``[S, H, D]`` into the zero-padded tile buffer."""
    import mlx.core as mx

    seq, heads, dim = x.shape
    if seq != geometry.total_seq_length:
        raise ValueError(f"VSA-H3 metadata was built for sequence length {geometry.total_seq_length}, got {seq}.")
    buf = mx.zeros((geometry.padded_length, heads, dim), dtype=x.dtype)
    buf = buf.at[mx.array(geometry.untile_combined_index)].add(x)
    return buf


def _untile_hidden(tiled, geometry: MiniMaxH3VSAGeometry):
    import mlx.core as mx

    return tiled[mx.array(geometry.untile_combined_index)]


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
        mx.contiguous(q.transpose(1, 0, 2))[None],
        mx.contiguous(k.transpose(1, 0, 2))[None],
        mx.contiguous(v.transpose(1, 0, 2))[None],
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
    bias = mx.where(mx.array(allow), mx.array(0.0, dtype=mx.float32), mx.array(-1e9, dtype=mx.float32))
    return mx.fast.scaled_dot_product_attention(
        mx.contiguous(q_tiled.transpose(1, 0, 2))[None],
        mx.contiguous(k_tiled.transpose(1, 0, 2))[None],
        mx.contiguous(v_tiled.transpose(1, 0, 2))[None],
        scale=scale,
        mask=bias[None],
    )[0].transpose(1, 0, 2)


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
    """Grouped gather + batched SDPA over video query tiles; prefix stays dense-equivalent.

    Full-sequence gather at 720p materializes tens of GiB of selected K/V, so
    query tiles are processed in memory-bounded chunks. This is the correctness
    baseline and the default ``auto`` path: on MLX 0.31.2 / M4 Max the fused
    SDPA gather is much faster than the scalar Metal kernel.
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
    video_flat = out.transpose(1, 2, 0, 3).reshape(n_q * tile_elems, heads, dim)
    prefix = q_tiled[:geometry.num_prefix_tiles * tile_elems]
    return mx.concatenate([prefix, video_flat], axis=0)


_METAL_KERNEL = None
_METAL_KERNEL_ERROR: str | None = None

# Cooperative K/V staging: 32 tokens x 128 dims x 4 bytes = 16 KiB of threadgroup
# memory. Every query thread in the tile reuses that staging buffer instead of
# reloading the same K/V rows from device memory. Do not early-return before
# barriers — padded query lanes still participate.
_METAL_SOURCE = """
    const int CHUNK = 32;
    const int MAXD = 128;
    threadgroup float smem[32 * 128];
    uint token = thread_position_in_grid.x;
    uint q_tile = thread_position_in_grid.y;
    uint head = thread_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint nthreads = threads_per_threadgroup.x;
    int D = meta_i[0];
    int TILE = meta_i[1];
    int S = meta_i[2];
    int k_max = meta_i[4];
    int n_prefix = meta_i[5];
    int n_video = meta_i[6];
    float scale = meta_f[0];
    if ((int)token >= TILE || (int)q_tile >= n_video || (int)head >= (int)q_shape[0]) {
        return;
    }
    int qt = n_prefix + (int)q_tile;
    int q_valid = vbs[qt];
    int q_base = (((int)head * S) + qt * TILE + (int)token) * D;
    float qreg[128];
    for (int d = 0; d < MAXD; d++) {
        qreg[d] = (d < D && (int)token < q_valid) ? float(q[q_base + d]) : 0.0f;
    }
    int nsel = block_num[(int)head * n_video + (int)q_tile];
    float m = -3.402823466e+38f;
    float lse = 0.0f;
    float acc[128];
    for (int d = 0; d < MAXD; d++) {
        acc[d] = 0.0f;
    }
    for (int s = 0; s < nsel; s++) {
        int kt = block_idx[(((int)head * n_video) + (int)q_tile) * k_max + s];
        int k_valid = vbs[kt];
        int tile_base = (((int)head * S) + kt * TILE) * D;
        for (int j0 = 0; j0 < TILE; j0 += CHUNK) {
            int n_elems = CHUNK * MAXD;
            for (uint i = tid; (int)i < n_elems; i += nthreads) {
                int tok = (int)i / MAXD;
                int d = (int)i - tok * MAXD;
                int gtok = j0 + tok;
                float val = 0.0f;
                if (gtok < k_valid && gtok < TILE && d < D) {
                    val = float(k[tile_base + gtok * D + d]);
                }
                smem[i] = val;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            float scores[32];
            float cmax = -3.402823466e+38f;
            for (int t = 0; t < CHUNK; t++) {
                int gtok = j0 + t;
                float score = 0.0f;
                if (gtok < k_valid && gtok < TILE) {
                    int sbase = t * MAXD;
                    for (int d = 0; d < D && d < MAXD; d++) {
                        score += qreg[d] * smem[sbase + d];
                    }
                    score *= scale;
                } else {
                    score = -3.402823466e+38f;
                }
                scores[t] = score;
                cmax = metal::max(cmax, score);
            }
            float m_new = metal::max(m, cmax);
            float alpha = metal::exp(m - m_new);
            lse *= alpha;
            for (int d = 0; d < D && d < MAXD; d++) {
                acc[d] *= alpha;
            }
            m = m_new;
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            for (uint i = tid; (int)i < n_elems; i += nthreads) {
                int tok = (int)i / MAXD;
                int d = (int)i - tok * MAXD;
                int gtok = j0 + tok;
                float val = 0.0f;
                if (gtok < k_valid && gtok < TILE && d < D) {
                    val = float(v[tile_base + gtok * D + d]);
                }
                smem[i] = val;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            for (int t = 0; t < CHUNK; t++) {
                float weight = ((j0 + t) < k_valid && (j0 + t) < TILE) ? metal::exp(scores[t] - m) : 0.0f;
                lse += weight;
                int sbase = t * MAXD;
                for (int d = 0; d < D && d < MAXD; d++) {
                    acc[d] += weight * smem[sbase + d];
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }
    }
    float denom = lse > 0.0f ? lse : 1.0f;
    for (int d = 0; d < D && d < MAXD; d++) {
        out[q_base + d] = ((int)token < q_valid) ? T(acc[d] / denom) : T(0);
    }
"""


def metal_kernel_available() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return hasattr(getattr(mx, "fast", None), "metal_kernel")


def _metal_kernel():
    global _METAL_KERNEL, _METAL_KERNEL_ERROR
    if _METAL_KERNEL is not None:
        return _METAL_KERNEL
    if _METAL_KERNEL_ERROR is not None:
        return None
    try:
        import mlx.core as mx

        if not hasattr(mx.fast, "metal_kernel"):
            _METAL_KERNEL_ERROR = "mx.fast.metal_kernel is not available in this MLX build"
            return None
        kwargs = {
            "name": "h3_vsa_block_sparse",
            "input_names": ["q", "k", "v", "block_idx", "block_num", "vbs", "meta_i", "meta_f"],
            "output_names": ["out"],
            "source": _METAL_SOURCE,
        }
        try:
            # MLX 0.32.2+ accepts compile_options (math_mode=safe).
            _METAL_KERNEL = mx.fast.metal_kernel(**kwargs, compile_options={"math_mode": "safe"})
        except TypeError:
            # MLX 0.31.2's metal_kernel() has no compile_options argument.
            _METAL_KERNEL = mx.fast.metal_kernel(**kwargs)
    except Exception as error:  # noqa: BLE001 - keep dense/reference usable
        _METAL_KERNEL_ERROR = str(error)
        _METAL_KERNEL = None
    return _METAL_KERNEL


def metal_supported(head_dim: int, tile_elems: int) -> bool:
    if not (0 < head_dim <= _METAL_MAX_HEAD_DIM and tile_elems in VSA_H3_TILE_SHAPES):
        return False
    return _metal_kernel() is not None


def _metal_block_sparse(
    q_tiled,
    k_tiled,
    v_tiled,
    block_idx,
    block_num,
    geometry: MiniMaxH3VSAGeometry,
    scale: float,
):
    import mlx.core as mx

    kernel = _metal_kernel()
    if kernel is None:
        raise RuntimeError(_METAL_KERNEL_ERROR or "Metal VSA kernel is unavailable")
    heads, dim = q_tiled.shape[1], q_tiled.shape[2]
    q = mx.contiguous(q_tiled.transpose(1, 0, 2))  # [H, S, D]
    k = mx.contiguous(k_tiled.transpose(1, 0, 2))
    v = mx.contiguous(v_tiled.transpose(1, 0, 2))
    meta_i = mx.array(
        [
            dim,
            geometry.tile_elems,
            geometry.padded_length,
            geometry.num_tiles,
            int(block_idx.shape[-1]),
            geometry.num_prefix_tiles,
            geometry.num_video_tiles,
        ],
        dtype=mx.int32,
    )
    meta_f = mx.array([scale], dtype=mx.float32)
    call_kwargs = {
        "inputs": [
            q,
            k,
            v,
            mx.contiguous(block_idx.astype(mx.int32)),
            mx.contiguous(block_num.astype(mx.int32)),
            mx.array(geometry.variable_block_sizes.astype(np.int32)),
            meta_i,
            meta_f,
        ],
        "template": [("T", q.dtype)],
        "grid": (geometry.tile_elems, geometry.num_video_tiles, heads),
        "threadgroup": (min(geometry.tile_elems, 256), 1, 1),
        "output_shapes": [q.shape],
        "output_dtypes": [q.dtype],
    }
    try:
        outputs = kernel(**call_kwargs, init_value=0)
    except TypeError:
        outputs = kernel(**call_kwargs)
    return outputs[0].transpose(1, 0, 2)


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
    if requested == "reference":
        return "reference"
    if requested == "metal":
        if not metal_supported(head_dim, tile_elems):
            reason = _METAL_KERNEL_ERROR or "Metal VSA kernel is unavailable for this shape"
            raise RuntimeError(reason)
        return "metal"
    # Measured on M4 Max / MLX 0.31.2 at 720p (H=56, D=128, tile=64, sparsity=0.9):
    # dense fused SDPA ~2.5s, chunked gather+SDPA ~2.9s, Metal kernel ~29s.
    # Prefer the gather reference unless the caller explicitly asked for Metal.
    return "reference"


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

    seq, heads, dim = query.shape
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
        chosen = resolve_impl(impl, dim, geometry.tile_elems)
        prefix_out = _dense_sdpa(query[:geometry.prefix_length], key, value, scale)
        if chosen == "metal":
            video_tiled = _metal_block_sparse(q_tiled, k_tiled, v_tiled, block_idx, block_num, geometry, scale)
            out_tiled = video_tiled
        elif geometry.num_tiles <= _REFERENCE_FULL_MASK_TILE_LIMIT:
            scores_np = np.array(scores, dtype=np.float32)
            mask = build_block_mask(
                scores_np,
                geometry.num_prefix_tiles,
                geometry.num_video_tiles,
                sparsity,
                exempt,
            )
            out_tiled = _reference_token_sdpa(q_tiled, k_tiled, v_tiled, mask, geometry, scale)
        else:
            out_tiled = _reference_gather_sdpa(q_tiled, k_tiled, v_tiled, block_idx, geometry, scale)
        out_tiled = out_tiled.astype(query.dtype)
        prefix_tiled = _tile_hidden(
            mx.concatenate([
                prefix_out,
                mx.zeros((seq - geometry.prefix_length, heads, dim), dtype=query.dtype),
            ],
                           axis=0),
            geometry,
        )
        # Prefix query tiles are dense; keep fused-SDPA prefix rows and sparse video tiles.
        n_prefix_pad = geometry.num_prefix_tiles * geometry.tile_elems
        out_tiled = mx.concatenate([prefix_tiled[:n_prefix_pad], out_tiled[n_prefix_pad:]], axis=0)

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
