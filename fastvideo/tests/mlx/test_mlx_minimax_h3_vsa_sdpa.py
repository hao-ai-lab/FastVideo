# SPDX-License-Identifier: Apache-2.0
"""Masked SDPA keeps H3's selected tiles and ragged-row semantics."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="Apple MLX is required")

from fastvideo.mlx_runtime import minimax_h3_vsa as vsa
from fastvideo.mlx_runtime import minimax_h3_vsa_sdpa as sdpa

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Fused MLX SDPA requires Metal")


def _selected_token_reference(q, k, value, indices, counts, geometry):
    """Independent token selection, followed by NumPy float64 softmax."""
    tile = geometry.tile_elems
    sizes = geometry.variable_block_sizes
    output = np.zeros_like(q, dtype=np.float64)
    for head in range(q.shape[1]):
        for query in range(geometry.num_video_tiles):
            query_tile = geometry.num_prefix_tiles + query
            query_rows = np.arange(query_tile * tile, query_tile * tile + sizes[query_tile])
            selected = indices[head, query, :counts[head, query]]
            if not len(selected):
                continue
            key_rows = np.concatenate([np.arange(index * tile, index * tile + sizes[index]) for index in selected])
            logits = q[query_rows, head].astype(np.float64) @ k[key_rows, head].astype(np.float64).T
            logits *= q.shape[-1]**-0.5
            probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            output[query_rows, head] = probabilities @ value[key_rows, head].astype(np.float64)
    return output


def _gathered_native_reference(q, k, value, indices, counts, geometry):
    """Native control with independently gathered valid keys and padded queries.

    Match the backend's query length: shorter ragged queries can select a
    different native arithmetic path. Query rows attend independently, so
    padding queries does not change the expected values of the valid rows.
    This control uses no GQA, key padding or attention mask.
    """
    output = np.zeros(q.shape, dtype=np.float32)
    tile = geometry.tile_elems
    sizes = geometry.variable_block_sizes
    for head in range(q.shape[1]):
        for query in range(geometry.num_video_tiles):
            query_tile = geometry.num_prefix_tiles + query
            selected = indices[head, query, :counts[head, query]]
            if not len(selected):
                continue
            query_start = query_tile * tile
            query_rows = np.arange(query_start, query_start + sizes[query_tile])
            key_rows = np.concatenate([np.arange(index * tile, index * tile + sizes[index]) for index in selected])
            result = mx.fast.scaled_dot_product_attention(
                q[query_start:query_start + tile, head][None, None],
                k[mx.array(key_rows), head][None, None],
                value[mx.array(key_rows), head][None, None],
                scale=q.shape[-1]**-0.5, force_fused=True)
            mx.eval(result)
            output[query_rows, head] = np.asarray(result[0, 0, :sizes[query_tile]].astype(mx.float32))
    return output


# MLX steel/attn/nax.h uses MPP matmul2d_descriptor(relaxed_precision=true),
# including float32 inputs. The MPP SDK explicitly permits an accuracy/performance
# tradeoff. On M5, the independent native gathered control and this backend both
# differ from NumPy by up to 7.45e-4 across these tile sizes. Matching their query
# lengths reduces native arithmetic differences without sharing key selection.
# Bound that native error separately from the stricter mask/layout control below.
@pytest.mark.parametrize("dtype_name,tolerance", [("float32", 1e-3), ("bfloat16", 0.02)])
@pytest.mark.parametrize("chunk_size", [1, 3, 8])
@pytest.mark.parametrize("tile_size", [64, 128])
def test_sparse_sdpa_matches_selected_valid_tokens(dtype_name, tolerance, chunk_size, tile_size):
    geometry = vsa.build_h3_segment_geometry((7, (1, 3, 5), 13, (3, 7, 11)), tile_size)
    rng = np.random.default_rng(9241)
    heads, dim = 3, 128
    shape = (geometry.padded_length, heads, dim)
    originals = [rng.normal(size=shape).astype(np.float32) * 0.6 for _ in range(3)]
    tile = geometry.tile_elems
    sizes = geometry.variable_block_sizes
    padding = np.concatenate([np.arange(index * tile + count, (index + 1) * tile)
                              for index, count in enumerate(sizes)])
    for index, array in enumerate(originals):
        array[padding] = 30.0 + index  # Missing padding masks cause large errors.
    width = min(5, geometry.num_tiles)
    indices = np.zeros((heads, geometry.num_video_tiles, width), dtype=np.int32)
    counts = np.empty((heads, geometry.num_video_tiles), dtype=np.int32)
    for head in range(heads):
        for query in range(geometry.num_video_tiles):
            count = 1 + (head + query) % width
            counts[head, query] = count
            indices[head, query, :count] = rng.permutation(geometry.num_tiles)[:count]
            unselected = np.setdiff1d(np.arange(geometry.num_tiles), indices[head, query, :count])
            indices[head, query, count:] = unselected[-1] if len(unselected) else 0
    counts[-1, -1] = 0
    q, k, value = [mx.array(item).astype(getattr(mx, dtype_name)) for item in originals]
    mx.eval(q, k, value)
    rounded = [np.asarray(item.astype(mx.float32)) for item in (q, k, value)]
    expected = _selected_token_reference(*rounded, indices, counts, geometry)
    actual = sdpa.sdpa_block_sparse(q, k, value, mx.array(indices), mx.array(counts), geometry,
                                    dim**-0.5, chunk_size=chunk_size)
    mx.eval(actual)
    got = np.asarray(actual.astype(mx.float32))
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, atol=tolerance, rtol=tolerance)
    if dtype_name == "float32":
        native = _gathered_native_reference(q, k, value, indices, counts, geometry)
        np.testing.assert_allclose(native, expected, atol=tolerance, rtol=tolerance)
        # This independent control removes all candidate mask construction,
        # reshaping and GQA, so a wrong selected tile or head cannot hide behind
        # the NAX kernel's looser float32 arithmetic.
        np.testing.assert_allclose(got, native, atol=1e-4, rtol=2e-4)
    np.testing.assert_array_equal(got[padding], 0)
    np.testing.assert_array_equal(got[:geometry.num_prefix_tiles * tile], 0)


def test_mask_memory_cap_splits_dispatch_without_changing_selected_tokens(monkeypatch):
    geometry = vsa.build_h3_segment_geometry((7, (1, 3, 5), 13, (3, 7, 11)), 128)
    rng = np.random.default_rng(116)
    heads, dim = 3, 128
    arrays = [rng.normal(size=(geometry.padded_length, heads, dim)).astype(np.float32) * 0.6
              for _ in range(3)]
    padding = np.concatenate([np.arange(tile * 128 + count, (tile + 1) * 128)
                              for tile, count in enumerate(geometry.variable_block_sizes)])
    for index, array in enumerate(arrays):
        array[padding] = 30.0 + index
    indices = np.empty((heads, geometry.num_video_tiles, 3), dtype=np.int32)
    for head in range(heads):
        for query in range(geometry.num_video_tiles):
            indices[head, query] = (head + query + np.array([0, 2, 4])) % geometry.num_tiles
    counts = np.full((heads, geometry.num_video_tiles), 3, dtype=np.int32)
    counts[-1, -1] = 0
    expected = _selected_token_reference(*arrays, indices, counts, geometry)
    q, k, value = [mx.array(array) for array in arrays]

    # A fixed tiny budget exercises the automatic cap without allocating a
    # production-sized mask. Observe actual native calls, not the chunk formula.
    cap_bytes = 8192
    monkeypatch.setattr(sdpa, "_MASK_CHUNK_MAX_BYTES", cap_bytes)
    dispatches = []
    native_sdpa = mx.fast.scaled_dot_product_attention

    def record_dispatch(query, key, value, **kwargs):
        dispatches.append({"mask_bytes": kwargs["mask"].nbytes,
                           "query_heads": query.shape[1]})
        return native_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", record_dispatch)
    actual = sdpa.sdpa_block_sparse(q, k, value, mx.array(indices), mx.array(counts), geometry,
                                    dim**-0.5, chunk_size=geometry.num_video_tiles)
    mx.eval(actual)
    assert len(dispatches) > 1
    assert all(0 < dispatch["mask_bytes"] <= cap_bytes for dispatch in dispatches)
    assert sum(dispatch["query_heads"] for dispatch in dispatches) == heads * geometry.num_video_tiles
    got = np.asarray(actual)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)
    np.testing.assert_array_equal(got[padding], 0)
    np.testing.assert_array_equal(got[:geometry.num_prefix_tiles * geometry.tile_elems], 0)


def _attention_inputs():
    geometry = vsa.build_h3_segment_geometry((7, (1, 3, 5), 13, (3, 7, 11)), 128)
    rng = np.random.default_rng(874)
    shape = (geometry.total_seq_length, 2, 128)
    tensors = [mx.array(rng.normal(size=shape).astype(np.float32) * 0.1) for _ in range(4)]
    return geometry, tensors


def test_explicit_sdpa_routes_without_changing_attention_selection(monkeypatch):
    geometry, (q, k, value, gate) = _attention_inputs()
    expected = vsa.h3_vsa_attention(q, k, value, geometry, sparsity=0.5, ref_keep_rate=0.5,
                                     gate_compress=gate, impl="reference")
    calls = []
    original = sdpa.sdpa_block_sparse

    def record(*args, **kwargs):
        calls.append((args[3].shape, args[4].shape))
        return original(*args, **kwargs)

    monkeypatch.setattr(sdpa, "sdpa_block_sparse", record)
    stats = vsa.MiniMaxH3VSAStats()
    actual = vsa.h3_vsa_attention(q, k, value, geometry, sparsity=0.5, ref_keep_rate=0.5,
                                  gate_compress=gate, impl="sdpa", stats=stats)
    mx.eval(actual, expected)
    assert len(calls) == 1
    assert calls[0][0][:2] == (2, geometry.num_video_tiles)
    assert calls[0][1] == (2, geometry.num_video_tiles)
    assert stats.impl == "sdpa"
    assert stats.dense_fallback_reason is None
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=2e-4)


def test_explicit_sdpa_propagates_kernel_failure(monkeypatch):
    geometry, (q, k, value, _) = _attention_inputs()

    def fail(*args, **kwargs):
        raise RuntimeError("test SDPA failure")

    monkeypatch.setattr(sdpa, "sdpa_block_sparse", fail)
    with pytest.raises(RuntimeError, match="test SDPA failure"):
        vsa.h3_vsa_attention(q, k, value, geometry, sparsity=0.5, ref_keep_rate=0.5, impl="sdpa")


def test_sdpa_is_explicit_and_auto_selection_stays_unchanged(monkeypatch):
    from fastvideo.mlx_runtime import minimax_h3_vsa_simd as simd

    monkeypatch.setattr(simd, "simd_kernel_available", lambda: True)
    assert vsa.resolve_impl("sdpa", 128, 128) == "sdpa"
    assert vsa.resolve_impl("auto", 128, 128) == "simd"


@pytest.mark.parametrize("head_dim,tile_size", [(64, 128), (128, 256)])
def test_explicit_sdpa_rejects_unsupported_shapes(head_dim, tile_size):
    with pytest.raises(ValueError):
        vsa.resolve_impl("sdpa", head_dim, tile_size)


def test_pipeline_rejects_sdpa_prefix_reuse_before_loading_components():
    from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline

    pipeline = object.__new__(MiniMaxH3MLXPipeline)
    with pytest.raises(ValueError, match="prefix|Prefix"):
        pipeline.generate("unused", output_path="unused.mp4", vsa_impl="sdpa", reuse_prefix_kv=True)


def test_sparse_sdpa_rejects_invalid_chunk_size():
    geometry = vsa.build_h3_segment_geometry((3, (1, 3, 5)), 128)
    array = mx.zeros((geometry.padded_length, 1, 128))
    indices = mx.zeros((1, geometry.num_video_tiles, 1), dtype=mx.int32)
    counts = mx.ones((1, geometry.num_video_tiles), dtype=mx.int32)
    with pytest.raises(ValueError, match="chunk"):
        sdpa.sdpa_block_sparse(array, array, array, indices, counts, geometry, 128**-0.5, chunk_size=0)
