# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""VSA tile masks through fused SDPA, sharing K/V with grouped-query attention.

This opt-in backend preserves selected blocks, ragged-row masks and the trained
compression branch. Unlike the SIMD kernel, fused SDPA visits masked key blocks;
on M5 its faster matrix operations can outweigh that additional work. Floating-
point accumulation differs, so this is not a bit-identical implementation.
"""

from __future__ import annotations

SDPA_TILE_SIZES = (64, 128)
SDPA_HEAD_DIM = 128
DEFAULT_QUERY_CHUNK = 128
_MASK_CHUNK_MAX_BYTES = 256 * 1024**2


def sdpa_block_sparse(q_tiled,
                      k_tiled,
                      v_tiled,
                      block_idx,
                      block_num,
                      geometry,
                      scale: float,
                      *,
                      chunk_size: int = DEFAULT_QUERY_CHUNK):
    """Return tiled [S_pad,H,D], with zero prefix rows and zero padded queries.

    Query heads are ordered (original head, query tile), so GQA maps each group
    of query tiles to its original K/V head. A tile's key mask broadcasts across
    its query rows. At most two mask chunks are in flight. Unsupported fused
    operations raise instead of silently allocating unfused attention scores.
    """
    import mlx.core as mx

    if chunk_size < 1:
        raise ValueError("SDPA VSA chunk_size must be positive")
    if q_tiled.ndim != 3 or k_tiled.shape != q_tiled.shape or v_tiled.shape != q_tiled.shape:
        raise ValueError("SDPA VSA requires matching [S_pad,H,D] Q/K/V shapes")
    padded, heads, dim = q_tiled.shape
    tile = geometry.tile_elems
    if tile not in SDPA_TILE_SIZES or dim != SDPA_HEAD_DIM:
        raise ValueError(f"SDPA VSA requires tile 64 or 128 and head dimension 128, got tile={tile}, dim={dim}")
    if padded != geometry.padded_length or heads < 1:
        raise ValueError("SDPA VSA Q/K/V shape does not match the tile geometry")
    if q_tiled.dtype != k_tiled.dtype or q_tiled.dtype != v_tiled.dtype:
        raise ValueError("SDPA VSA requires matching Q/K/V dtypes")
    n_tiles = geometry.num_tiles
    n_video = geometry.num_video_tiles
    n_prefix = geometry.num_prefix_tiles
    if (block_idx.ndim != 3 or block_idx.shape[:2] != (heads, n_video) or block_num.shape != (heads, n_video)):
        raise ValueError("SDPA VSA routing shape does not match the query head/tile geometry")

    query_chunk = min(chunk_size, max(1, _MASK_CHUNK_MAX_BYTES // max(heads * padded, 1)))
    q_video = q_tiled.reshape(n_tiles, tile, heads, dim).transpose(2, 0, 1, 3)[:, n_prefix:]
    # Fused full attention accepts these strided views: the D axis is contiguous.
    k_shared = k_tiled.transpose(1, 0, 2)[None]
    v_shared = v_tiled.transpose(1, 0, 2)[None]
    sizes = mx.array(geometry.variable_block_sizes, dtype=mx.int32)
    row = mx.arange(tile, dtype=mx.int32)
    key_valid = row[None, :] < sizes[:, None]
    head_index = mx.arange(heads, dtype=mx.int32)[:, None, None]
    selected_slot = mx.arange(block_idx.shape[-1], dtype=mx.int32)[None, None, :]
    outputs = []
    for start in range(0, n_video, query_chunk):
        end = min(start + query_chunk, n_video)
        count = end - start
        indices = block_idx[:, start:end].astype(mx.int32)
        counts = block_num[:, start:end]
        # Send inactive slots to a sentinel column that is discarded. Every
        # scatter writes True, so duplicate indices cannot erase a valid key.
        indices = mx.where(selected_slot < counts[..., None], indices, n_tiles)
        keep = mx.zeros((heads, count, n_tiles + 1), dtype=mx.bool_)
        tile_index = mx.arange(count, dtype=mx.int32)[None, :, None]
        keep[head_index, tile_index, indices] = True
        mask = (keep[:, :, :n_tiles, None] & key_valid[None, None]).reshape(1, heads * count, 1, padded)
        query = mx.contiguous(q_video[:, start:end].reshape(1, heads * count, tile, dim))
        output = mx.fast.scaled_dot_product_attention(query,
                                                      k_shared,
                                                      v_shared,
                                                      scale=scale,
                                                      mask=mask,
                                                      force_fused=True)
        output = output.reshape(heads, count, tile, dim)
        valid_query = row[None, :] < sizes[n_prefix + start:n_prefix + end, None]
        output = mx.where(valid_query[None, :, :, None] & (counts[..., None, None] > 0), output, 0)
        mx.async_eval(output)
        outputs.append(output)
        if len(outputs) > 1:
            mx.eval(outputs[-2])
    if not outputs:
        return mx.zeros_like(q_tiled)
    mx.eval(outputs[-1])
    video = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
    video = video.transpose(1, 2, 0, 3).reshape(n_video * tile, heads, dim)
    return mx.concatenate((mx.zeros((n_prefix * tile, heads, dim), dtype=q_tiled.dtype), video), axis=0)
