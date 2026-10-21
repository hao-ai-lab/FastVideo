# SPDX-License-Identifier: Apache-2.0
"""Changing the gather memory budget must preserve selected-tile attention."""
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
import fastvideo.mlx_runtime.minimax_h3_vsa as vsa


def test_gather_budget_preserves_attention_with_partial_tiles(monkeypatch):
    geom = vsa.build_h3_tile_geometry((7, 67), (6, 8, 8), 64)
    rng = np.random.default_rng(2026)
    shape = (geom.padded_length, 2, 128)
    q, k, value = [mx.array(rng.normal(size=shape).astype(np.float32)).astype(mx.bfloat16) for _ in range(3)]
    # Reverse video order also exercises the selected-key order, not a dense mask.
    selected = np.array([0, 1, geom.num_tiles - 1, geom.num_prefix_tiles], dtype=np.int32)
    idx = mx.array(np.broadcast_to(selected, (2, geom.num_video_tiles, selected.size)).copy())
    monkeypatch.setattr(vsa, '_reference_gather_target_bytes', lambda: 2 * 1024**3)
    expected = vsa._reference_gather_sdpa(q, k, value, idx, geom, 128**-0.5)
    mx.eval(expected)
    monkeypatch.setattr(vsa, '_reference_gather_target_bytes', lambda: 1)
    actual = vsa._reference_gather_sdpa(q, k, value, idx, geom, 128**-0.5)
    mx.eval(actual)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)))


def test_row_take_gather_matches_advanced_indexing():
    rng = np.random.default_rng(7)
    heads, n_tiles, tile, dim = 3, 5, 16, 8
    k = mx.array(rng.normal(size=(heads, n_tiles, tile, dim)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.normal(size=(heads, n_tiles, tile, dim)).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(rng.integers(0, n_tiles, size=(heads, 4, 3)).astype(np.int32))
    got_k, got_v = vsa._gather_selected_kv(vsa._tile_rows(k), vsa._tile_rows(v), idx, n_tiles, tile)
    heads_ix = mx.arange(heads)[:, None, None]
    want_k = k[heads_ix, idx].reshape(heads, 4, 3 * tile, dim)
    want_v = v[heads_ix, idx].reshape(heads, 4, 3 * tile, dim)
    np.testing.assert_array_equal(np.array(got_k.astype(mx.float32)), np.array(want_k.astype(mx.float32)))
    np.testing.assert_array_equal(np.array(got_v.astype(mx.float32)), np.array(want_v.astype(mx.float32)))


def test_compiled_rotary_matches_uncompiled_fp32_math():
    from fastvideo.mlx_runtime.minimax_h3 import apply_h3_rotary, rope_cos_sin

    rng = np.random.default_rng(11)
    x = mx.array(rng.normal(size=(40, 4, 128)).astype(np.float32)).astype(mx.bfloat16)
    cos, sin = rope_cos_sin(mx.array(rng.uniform(0, 50, size=(40, 3)).astype(np.float32)), 16, 10000.0)
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim].astype(mx.float32)
    first, second = mx.split(x_rot, 2, axis=-1)
    rotated = mx.concatenate([-second, first], axis=-1)
    out = (x_rot * cos[:, None, :] + rotated * sin[:, None, :]).astype(x.dtype)
    expected = mx.concatenate([out, x[..., rotary_dim:]], axis=-1)
    np.testing.assert_array_equal(np.array(apply_h3_rotary(x, cos, sin).astype(mx.float32)),
                                  np.array(expected.astype(mx.float32)))
