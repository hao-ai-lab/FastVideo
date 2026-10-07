# SPDX-License-Identifier: Apache-2.0
"""FASTVIDEO_H3_VSA_HEADS_FIRST_TILE must hand VSA-H3 the same tiled tensor, bit for bit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.attention.backends import video_sparse_attn_h3 as vsa
from fastvideo.attention.backends.video_sparse_attn_h3_scatter import (HAVE_TRITON, scatter_rows_heads_first,
                                                                       supports_heads_first_scatter)


def _require_cuda_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("the heads-first tile scatter requires CUDA")
    if not HAVE_TRITON:
        pytest.skip("the heads-first tile scatter requires Triton")


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _metadata(seq: int, n_tiles: int, tile_elems: int) -> SimpleNamespace:
    slots = torch.randperm(n_tiles * tile_elems, device="cuda")[:seq].sort().values
    sizes = torch.bincount(slots // tile_elems, minlength=n_tiles)
    return SimpleNamespace(total_seq_length=seq,
                           variable_block_sizes=sizes,
                           tile_elems=tile_elems,
                           untile_combined_index=slots,
                           tile_buf_holder=vsa._MiniMaxH3VSATileBufferHolder())


def _impl() -> vsa.MiniMaxH3VSAImpl:
    impl = object.__new__(vsa.MiniMaxH3VSAImpl)
    impl._regional_compile_sm100a_enabled = None
    return impl


@pytest.fixture
def sm100a():
    with envs.FASTVIDEO_VSA_SM100A.override(True):
        yield


@pytest.mark.gpu
def test_scatter_rows_heads_first_matches_indexed_assignment() -> None:
    _require_cuda_triton()
    torch.manual_seed(0)
    seq, heads, dim, slots_total = 5000, 12, 128, 6144
    x = torch.randn(1, seq, heads, dim, device="cuda").to(torch.bfloat16)
    slots = torch.randperm(slots_total, device="cuda")[:seq]
    assert supports_heads_first_scatter(x, slots)
    expected = torch.zeros(1, slots_total, heads, dim, device="cuda", dtype=x.dtype)
    expected[:, slots] = x
    buffer = torch.zeros(1, heads, slots_total, dim, device="cuda", dtype=x.dtype)
    scatter_rows_heads_first(x, slots, buffer)
    assert _same_bits(expected, buffer.transpose(1, 2))


@pytest.mark.gpu
@pytest.mark.parametrize(("tile_elems", "n_tiles"), [(128, 37), (128, 40), (64, 41)])
def test_heads_first_tile_equals_the_default_tile(sm100a, tile_elems: int, n_tiles: int) -> None:
    _require_cuda_triton()
    torch.manual_seed(1)
    meta = _metadata(seq=n_tiles * tile_elems - 300, n_tiles=n_tiles, tile_elems=tile_elems)
    x = torch.randn(1, meta.total_seq_length, 8, 128, device="cuda").to(torch.bfloat16)
    impl = _impl()
    with torch.inference_mode():
        with envs.FASTVIDEO_H3_VSA_HEADS_FIRST_TILE.override(False):
            default = impl.tile(x, meta).clone()
        with envs.FASTVIDEO_H3_VSA_HEADS_FIRST_TILE.override(True):
            heads_first = impl.tile(x, meta)
        assert meta.tile_buf_holder.heads_first_buffer is not None
        assert heads_first.transpose(1, 2).is_contiguous()
        assert _same_bits(default, heads_first)
        # The tile-score pooling reads the logical prefix of the view directly
        # (forward() drops the sm100a pairing tile); its FP32 sums must not move.
        logical = n_tiles * tile_elems
        assert _same_bits(vsa._pool_tiles(default[:, :logical], meta.variable_block_sizes, tile_elems),
                          vsa._pool_tiles(heads_first[:, :logical], meta.variable_block_sizes, tile_elems))
        # A new geometry on the reused buffer leaves no stale row behind.
        other = _metadata(seq=meta.total_seq_length, n_tiles=n_tiles, tile_elems=tile_elems)
        other.tile_buf_holder = meta.tile_buf_holder
        with envs.FASTVIDEO_H3_VSA_HEADS_FIRST_TILE.override(False):
            default = impl.tile(x, other).clone()
        with envs.FASTVIDEO_H3_VSA_HEADS_FIRST_TILE.override(True):
            assert _same_bits(default, impl.tile(x, other))
