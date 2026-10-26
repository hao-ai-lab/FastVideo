# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""VSA attention for the target rows only (prefix K/V reuse).

Queries are the packed tail ``[target audio | target video]``; keys and values
cover the whole packed sequence (cached frozen rows followed by the fresh
target rows). Tile geometry, pooled scoring, top-k routing and the compression
gate are the ones ``h3_vsa_attention`` uses, evaluated only for the target
query tiles, so given identical K/V a target row selects exactly the tiles the
full path would have given it.
"""

from __future__ import annotations

from fastvideo.mlx_runtime.minimax_h3_prefix_kv import TargetQueryPlan
from fastvideo.mlx_runtime.minimax_h3_vsa import (
    MiniMaxH3VSAGeometry,
    MiniMaxH3VSAStats,
    VSAImpl,
    _block_indices_from_scores,
    _dense_sdpa,
    _pool_tiles,
    _reference_gather_sdpa,
    _tile_hidden,
    compute_topk,
    region_sparsities,
    resolve_impl,
)


def h3_vsa_attention_target(
    query,
    key,
    value,
    geometry: MiniMaxH3VSAGeometry,
    plan: TargetQueryPlan,
    *,
    sparsity: float,
    exempt: bool = True,
    gate_compress=None,
    impl: VSAImpl = "auto",
    stats: MiniMaxH3VSAStats | None = None,
    ref_keep_rate: float | None = None,
):
    """query/gate_compress: [T, H, D] target rows; key/value: [S, H, D] packed. Returns [T, H, D]."""
    if impl in ("sdpa", "nax"):
        raise ValueError(
            f"{impl.upper()} VSA does not support prefix K/V reuse; disable reuse or select another backend")
    import mlx.core as mx

    _, heads, dim = query.shape
    scale = dim**-0.5
    tile = geometry.tile_elems
    if stats is not None:
        stats.configured_sparsity = sparsity
        stats.layer_sparsity = sparsity
        stats.tile_size = tile
        stats.prefix_mode = "exempt" if exempt else "compete"
        stats.num_prefix_tiles = geometry.num_prefix_tiles
        stats.num_video_tiles = geometry.num_video_tiles

    if sparsity <= 0.0 and gate_compress is None:
        if stats is not None:
            stats.impl = "dense"
            stats.achieved_sparsity = 0.0
            stats.video_keep = geometry.num_video_tiles
        return _dense_sdpa(query, key, value, scale)

    pad_row = mx.zeros((1, heads, dim), dtype=query.dtype)
    num_query_tiles = int(plan.query_tiles.size)
    query_block_sizes = geometry.variable_block_sizes[plan.query_tiles]
    q_tiled = mx.concatenate([query, pad_row])[plan.tile_gather]
    k_tiled = _tile_hidden(key, geometry)
    v_tiled = _tile_hidden(value, geometry)
    q_pooled = _pool_tiles(q_tiled, query_block_sizes, tile)
    k_pooled = _pool_tiles(k_tiled, geometry.variable_block_sizes, tile)
    scores = (q_pooled @ k_pooled.transpose(0, 2, 1)) / (dim**0.5)  # [H, query tiles, all tiles]

    spans = geometry.video_tile_spans
    span_sparsities = region_sparsities(len(spans), sparsity, ref_keep_rate)
    k_vid = sum(compute_topk(sp, end - start) for (start, end), sp in zip(spans, span_sparsities, strict=True))
    if stats is not None:
        stats.video_keep = k_vid if sparsity > 0.0 else geometry.num_video_tiles
        stats.achieved_sparsity = (1.0 - (k_vid / geometry.num_video_tiles) if geometry.num_video_tiles else 0.0)

    if sparsity <= 0.0:
        rows_out = _dense_sdpa(query, key, value, scale)
        out_tiled = mx.concatenate([rows_out, pad_row])[plan.tile_gather]
        chosen = "dense"
    else:
        block_idx, _ = _block_indices_from_scores(
            scores[:, plan.num_audio_tiles:, :],
            geometry.num_prefix_tiles,
            geometry.num_video_tiles,
            sparsity,
            exempt,
            spans,
            span_sparsities,
            query_scores_only=True,
        )
        if stats is not None and not exempt:
            stats.video_keep = float(mx.mean(mx.sum(block_idx >= geometry.num_prefix_tiles, axis=-1)).item())
            stats.achieved_sparsity = 1.0 - stats.video_keep / geometry.num_video_tiles
        chosen = resolve_impl(impl, dim, tile)
        if chosen == "simd":
            # The SIMD kernel launches over every video tile; the target-only path uses the gather backend.
            chosen = "reference"
            if stats is not None:
                stats.dense_fallback_reason = "prefix K/V reuse uses the reference VSA backend"
        q_video = q_tiled[plan.num_audio_tiles * tile:].reshape(-1, tile, heads, dim).transpose(2, 0, 1, 3)
        video_tiled = _reference_gather_sdpa(q_tiled,
                                             k_tiled,
                                             v_tiled,
                                             block_idx,
                                             geometry,
                                             scale,
                                             q_video_tiles=q_video).astype(query.dtype)
        parts = []
        if plan.num_audio_tiles:
            audio_out = _dense_sdpa(query[:plan.num_audio_rows], key, value, scale)
            parts.append(mx.concatenate([audio_out, pad_row])[plan.audio_tile_gather])
        parts.append(video_tiled)
        out_tiled = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    if gate_compress is not None:
        gate_tiled = mx.concatenate([gate_compress, pad_row.astype(gate_compress.dtype)])[plan.tile_gather]
        v_pooled = _pool_tiles(v_tiled, geometry.variable_block_sizes, tile)
        out_c = (mx.softmax(scores, axis=-1) @ v_pooled).transpose(1, 0, 2)  # [query tiles, H, D]
        gate_view = gate_tiled.reshape(num_query_tiles, tile, heads, dim)
        out_tiled = out_tiled + (out_c[:, None, :, :] * gate_view).reshape(num_query_tiles * tile, heads, dim).astype(
            out_tiled.dtype)

    if stats is not None:
        stats.impl = chosen if sparsity > 0.0 else "dense"
    return out_tiled[plan.out_slot]
