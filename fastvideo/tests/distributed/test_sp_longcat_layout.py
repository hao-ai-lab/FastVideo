# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for LongCatSPLayout (LongCat BSA sequence-parallel sharding).

These pin the layout math the multi-GPU BSA-refinement SP path depends on
(fastvideo/models/dits/longcat.py ``LongCatSPLayout``): the ``split_h``/``split_w``
factors, the rank<->(row, col) mapping in ``shard``, and the global<->rank-major
permutes in ``rank_to_global``/``global_to_rank``. A future edit to any of these
that breaks the single-GPU-equivalent token ordering would otherwise ship with no
signal and surface only as a wrong or crashing video on multi-GPU hardware.

Single-process and CPU-resident (plain tensor reshape/permute; no torchrun, no
GPU compute). It imports ``fastvideo.models.dits.longcat``, which pulls in the
standard FastVideo dev/GPU stack (torch + triton via the BSA interface) -- the
same import the existing ``test_sp_*.py`` model tests perform.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.models.dits.longcat import LongCatSPLayout


def _case(t: int, h: int, w: int, size: int):
    return pytest.param((t, h, w), size, id=f"grid{t}x{h}x{w}-sp{size}")


# Grids whose token dims divide the SP split factors (h % split_h == 0 and
# w % split_w == 0), across the SP sizes LongCat refinement is expected to run.
_VALID = [
    _case(4, 8, 8, 2),
    _case(4, 8, 8, 4),
    _case(4, 16, 16, 8),
    _case(4, 8, 16, 4),
    _case(8, 4, 8, 2),
    _case(4, 32, 32, 16),
    _case(2, 4, 4, 2),
]


@pytest.mark.parametrize("grid,size", _VALID)
def test_rank_global_roundtrip(grid, size):
    t, h, w = grid
    layout = LongCatSPLayout(grid, size)
    n = t * h * w
    x = torch.arange(n).reshape(1, n, 1)
    # global -> rank-major -> global is the identity
    assert torch.equal(layout.rank_to_global(layout.global_to_rank(x)), x)
    # rank-major -> global -> rank-major is the identity
    y = layout.global_to_rank(x)
    assert torch.equal(layout.global_to_rank(layout.rank_to_global(y)), y)


@pytest.mark.parametrize("grid,size", _VALID)
def test_shard_blocks_reconstruct_global(grid, size):
    t, h, w = grid
    layout = LongCatSPLayout(grid, size)
    n = t * h * w
    x = torch.arange(n).reshape(1, n, 1)
    shards = [layout.shard(x, rank) for rank in range(size)]
    for s in shards:
        assert s.shape == (1, n // size, 1)
    # Concatenating per-rank shards in rank order and mapping back to global
    # reproduces the input exactly (shard is consistent with global_to_rank).
    cat = torch.cat(shards, dim=1)
    assert torch.equal(layout.rank_to_global(cat), x)


def test_split_factors():
    assert (LongCatSPLayout((4, 8, 8), 2).split_h, LongCatSPLayout((4, 8, 8), 2).split_w) == (1, 2)
    assert (LongCatSPLayout((4, 8, 8), 4).split_h, LongCatSPLayout((4, 8, 8), 4).split_w) == (2, 2)
    assert (LongCatSPLayout((4, 8, 8), 8).split_h, LongCatSPLayout((4, 8, 8), 8).split_w) == (2, 4)
    assert (LongCatSPLayout((4, 32, 32), 16).split_h, LongCatSPLayout((4, 32, 32), 16).split_w) == (4, 4)
    assert (LongCatSPLayout((4, 32, 32), 32).split_h, LongCatSPLayout((4, 32, 32), 32).split_w) == (4, 8)


def test_rejects_unsplittable_grid():
    # sp=32 -> split_h=4, split_w=8; w=6 is not divisible by split_w=8
    with pytest.raises(ValueError):
        LongCatSPLayout((4, 8, 6), 32)
    # sp=4 -> split_h=2, split_w=2; h=3 is not divisible by split_h=2
    with pytest.raises(ValueError):
        LongCatSPLayout((4, 3, 4), 4)


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        LongCatSPLayout((4, 8, 8), 0)  # size < 1
    with pytest.raises(ValueError):
        LongCatSPLayout((4, 8), 4)  # len(grid) != 3