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
