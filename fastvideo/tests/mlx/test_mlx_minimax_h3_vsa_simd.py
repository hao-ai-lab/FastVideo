# SPDX-License-Identifier: Apache-2.0
"""Parity of the SIMD-group block-sparse VSA kernel (tiles 64 and 128).

The reference impl (gather + SDPA, itself CUDA-parity tested) and an
independent fp32 numpy oracle are the baselines. Requires Metal execution.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 VSA tests")

from fastvideo.mlx_runtime import minimax_h3_vsa as vsa  # noqa: E402
from fastvideo.mlx_runtime import minimax_h3_vsa_simd as simd  # noqa: E402

pytestmark = pytest.mark.skipif(not simd.simd_kernel_available(), reason="SIMD-group VSA kernel is not available")

_DIM = 128
# (name, segments). Ints are dense rows, tuples are video token grids; the last tuple is the target video.
_LAYOUTS = {
    # Image-only request: odd prefix chunks and a ragged target grid.
    "image_only": (1, 127, 129, 63, (5, 6, 7)),
    "image_only_exact_tiles": (64, 128, (4, 8, 8)),
    # Video-reference request: dense text/image/audio around a reference video region.
    "video_ref": (9, 70, (5, 6, 7), 13, (7, 9, 11)),
    "video_ref_two": (3, (6, 5, 9), 65, (3, 3, 3), 31, (6, 10, 5)),
    # Every tile of the video is ragged.
    "tiny_grid": (5, (3, 3, 3)),
}


def _rand_qkv(n: int, heads: int, seed: int, amp: float = 1.0):
    rng = np.random.default_rng(seed)
    return [mx.array(rng.standard_normal((n, heads, _DIM)).astype(np.float32) * amp).astype(mx.bfloat16)
            for _ in range(3)]


def _err(a, b):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    rel = float(mx.linalg.norm(a32 - b32) / mx.linalg.norm(b32))
    return rel, float(mx.abs(a32 - b32).max())


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("sparsity", [0.5, 0.9])
def test_simd_matches_reference(tile, layout, heads, sparsity) -> None:
    geo = vsa.build_h3_segment_geometry(_LAYOUTS[layout], tile)
    q, k, v = _rand_qkv(geo.total_seq_length, heads, seed=heads * 7 + tile)
    kwargs = dict(sparsity=sparsity, exempt=True, ref_keep_rate=0.3)
    ref = vsa.h3_vsa_attention(q, k, v, geo, impl="reference", **kwargs)
    stats = vsa.MiniMaxH3VSAStats()
    got = vsa.h3_vsa_attention(q, k, v, geo, impl="simd", stats=stats, **kwargs)
    mx.eval(ref, got)
    assert stats.impl == "simd" and stats.dense_fallback_reason is None
    assert got.dtype == mx.bfloat16
    rel, worst = _err(got, ref)
    assert rel < 1e-2 and worst < 5e-2, (rel, worst)


@pytest.mark.parametrize("tile", [64, 128])
def test_simd_matches_reference_with_sharp_softmax(tile) -> None:
    """Large logits make a mis-masked or mis-normalised key visible."""
    geo = vsa.build_h3_segment_geometry(_LAYOUTS["video_ref"], tile)
    q, k, v = _rand_qkv(geo.total_seq_length, 2, seed=3, amp=3.0)
    ref = vsa.h3_vsa_attention(q, k, v, geo, sparsity=0.75, impl="reference", ref_keep_rate=0.5)
    got = vsa.h3_vsa_attention(q, k, v, geo, sparsity=0.75, impl="simd", ref_keep_rate=0.5)
    rel, worst = _err(got, ref)
    assert rel < 2e-2 and worst < 0.25, (rel, worst)


def _oracle(q, k, v, block_idx, block_num, geo, scale):
    """fp32 attention of each video query tile over the valid rows of its selected key tiles."""
    t = geo.tile_elems
    sizes = geo.variable_block_sizes
    q, k, v = (np.asarray(x.astype(mx.float32)).reshape(geo.num_tiles, t, -1, _DIM) for x in (q, k, v))
    out = np.zeros_like(q)
    for head in range(q.shape[2]):
        for qi in range(geo.num_video_tiles):
            qt = geo.num_prefix_tiles + qi
            tiles = np.asarray(block_idx)[head, qi, :int(np.asarray(block_num)[head, qi])]
            keys = np.concatenate([k[kt, :sizes[kt], head] for kt in tiles])
            vals = np.concatenate([v[kt, :sizes[kt], head] for kt in tiles])
            s = q[qt, :sizes[qt], head] @ keys.T * scale
            p = np.exp(s - s.max(-1, keepdims=True))
            out[qt, :sizes[qt], head] = (p / p.sum(-1, keepdims=True)) @ vals
    return out


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("garbage_padding", [False, True])
def test_partial_tiles_match_oracle_and_ignore_padding(tile, garbage_padding) -> None:
    """Ragged query and key tiles: padded keys never contribute, padded query rows are zero.

    This targets the old "partial-tile correctness issue": every video tile of
    a (5, 5, 5) grid is ragged, tile selections are shuffled so ragged key tiles
    come first/last, and per-query ``block_num`` is shorter than ``k_max``.
    """
    geo = vsa.build_h3_segment_geometry((7, (5, 5, 5)), tile)
    sizes = geo.variable_block_sizes
    assert (sizes < tile).any() and (sizes[geo.num_prefix_tiles:] < tile).any()
    heads = 2
    q, k, v = _rand_qkv(geo.total_seq_length, heads, seed=11)
    q_t, k_t, v_t = (vsa._tile_hidden(x, geo) for x in (q, k, v))
    if garbage_padding:
        pad = np.ones((geo.num_tiles, tile), dtype=bool)
        for i, size in enumerate(sizes):
            pad[i, :size] = False
        pad = mx.array(pad.reshape(-1))[:, None, None]
        junk = mx.full(k_t.shape, 50.0, dtype=mx.bfloat16)
        q_t, k_t, v_t = (mx.where(pad, junk, x) for x in (q_t, k_t, v_t))
    rng = np.random.default_rng(0)
    n_tiles, n_video, k_max = geo.num_tiles, geo.num_video_tiles, min(geo.num_tiles, 4)
    idx = np.stack([[rng.permutation(n_tiles)[:k_max] for _ in range(n_video)] for _ in range(heads)]).astype(np.int32)
    num = rng.integers(1, k_max + 1, size=(heads, n_video)).astype(np.int32)
    scale = _DIM**-0.5
    got = simd._launch_simd_block_sparse(simd._simd_kernel(), q_t, k_t, v_t, mx.array(idx), mx.array(num), geo, scale)
    mx.eval(got)
    got = np.asarray(got.astype(mx.float32)).reshape(n_tiles, tile, heads, _DIM)
    # The oracle reads only valid rows, so it can use the garbage-padded tensors directly.
    want = _oracle(q_t, k_t, v_t, idx, num, geo, scale)
    np.testing.assert_allclose(got, want, atol=3e-2, rtol=3e-2)
    for qt in range(geo.num_prefix_tiles, n_tiles):
        assert not got[qt, sizes[qt]:].any(), "padded query rows must be zero"


@pytest.mark.parametrize("tile", [64, 128])
def test_simd_matches_dense_when_every_tile_is_selected(tile) -> None:
    geo = vsa.build_h3_segment_geometry((10, 200, (6, 7, 9)), tile)
    q, k, v = _rand_qkv(geo.total_seq_length, 2, seed=5)
    dense = vsa._dense_sdpa(q, k, v, _DIM**-0.5)
    got = vsa.h3_vsa_attention(q, k, v, geo, sparsity=1e-6, impl="simd")
    rel, worst = _err(got, dense)
    assert rel < 1e-2 and worst < 5e-2, (rel, worst)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("tile", [64, 128])
def test_simd_supports_every_input_dtype(dtype, tile) -> None:
    geo = vsa.build_h3_segment_geometry((20, (4, 6, 9)), tile)
    q, k, v = (x.astype(dtype) for x in _rand_qkv(geo.total_seq_length, 2, seed=2))
    ref = vsa.h3_vsa_attention(q, k, v, geo, sparsity=0.5, impl="reference")
    got = vsa.h3_vsa_attention(q, k, v, geo, sparsity=0.5, impl="simd")
    assert got.dtype == dtype
    rel, _ = _err(got, ref)
    assert rel < 1e-2


def test_auto_resolves_tile_128_to_simd_and_reference_overrides() -> None:
    assert vsa.resolve_impl("auto", _DIM, 128) == "simd"
    assert vsa.resolve_impl("reference", _DIM, 128) == "reference"
    assert vsa.resolve_impl("auto", _DIM, 256) == "reference"
    assert vsa.resolve_impl("simd", _DIM, 64) == "simd"
    assert vsa.resolve_impl("simd", 64, 128) == "reference"


@pytest.mark.parametrize("tile", simd.SIMD_TILE_SIZES)
def test_startup_probe_inputs_cover_the_padded_geometry(tile) -> None:
    """The capability probe must size Q/K/V and routing like a real launch, or it reads/writes out of bounds."""
    q, indices, counts, geometry = simd._probe_inputs(tile, mx.float32)
    assert q.shape[0] == geometry.padded_length
    assert indices.shape[1] == counts.shape[1] == geometry.num_video_tiles
    assert int(mx.max(indices).item()) < geometry.num_tiles
