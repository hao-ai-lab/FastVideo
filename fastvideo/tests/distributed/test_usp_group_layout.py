# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the USP (Ring x Ulysses) 2D rank-mesh layout.

``_build_usp_subgroup_ranks`` is pure rank arithmetic (no process group
creation), so it is tested directly here rather than through a multi-process
distributed test. See ``fastvideo/tests/distributed/test_ring_attention.py``
for the distributed correctness test of the Ring kernel itself.
"""

from __future__ import annotations

from fastvideo.distributed.parallel_state import _build_usp_subgroup_ranks


def test_single_replica_hybrid_layout() -> None:
    """ring_size=2, ulysses_size=2: Ulysses groups are contiguous blocks,
    Ring groups are strided across those blocks."""
    ulysses_group_ranks, ring_group_ranks = _build_usp_subgroup_ranks(
        sp_group_ranks=[[0, 1, 2, 3]],
        ring_size=2,
        ulysses_size=2,
    )

    assert ulysses_group_ranks == [[0, 1], [2, 3]]
    assert ring_group_ranks == [[0, 2], [1, 3]]


def test_multiple_sp_replicas_are_independent() -> None:
    """Each SP replica gets its own independent set of subgroups."""
    ulysses_group_ranks, ring_group_ranks = _build_usp_subgroup_ranks(
        sp_group_ranks=[[0, 1, 2, 3], [4, 5, 6, 7]],
        ring_size=2,
        ulysses_size=2,
    )

    assert ulysses_group_ranks == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert ring_group_ranks == [[0, 2], [1, 3], [4, 6], [5, 7]]


def test_pure_ring_degenerate_shape() -> None:
    """ulysses_size=1: one Ulysses group per rank (singletons), one Ring
    group spanning the whole SP replica."""
    ulysses_group_ranks, ring_group_ranks = _build_usp_subgroup_ranks(
        sp_group_ranks=[[0, 1, 2, 3]],
        ring_size=4,
        ulysses_size=1,
    )

    assert ulysses_group_ranks == [[0], [1], [2], [3]]
    assert ring_group_ranks == [[0, 1, 2, 3]]


def test_pure_ulysses_degenerate_shape() -> None:
    """ring_size=1: one Ulysses group spanning the whole SP replica, one
    singleton Ring group per rank."""
    ulysses_group_ranks, ring_group_ranks = _build_usp_subgroup_ranks(
        sp_group_ranks=[[0, 1, 2, 3]],
        ring_size=1,
        ulysses_size=4,
    )

    assert ulysses_group_ranks == [[0, 1, 2, 3]]
    assert ring_group_ranks == [[0], [1], [2], [3]]


def test_every_rank_appears_exactly_once_in_each_dimension() -> None:
    """For a larger, non-power-of-two mesh, every global rank must appear in
    exactly one Ulysses group and exactly one Ring group (the mesh must be a
    strict partition of the SP replica's ranks along each axis)."""
    sp_ranks = list(range(12))
    ring_size, ulysses_size = 3, 4

    ulysses_group_ranks, ring_group_ranks = _build_usp_subgroup_ranks(
        sp_group_ranks=[sp_ranks],
        ring_size=ring_size,
        ulysses_size=ulysses_size,
    )

    assert len(ulysses_group_ranks) == ring_size
    assert len(ring_group_ranks) == ulysses_size
    assert all(len(g) == ulysses_size for g in ulysses_group_ranks)
    assert all(len(g) == ring_size for g in ring_group_ranks)

    assert sorted(r for g in ulysses_group_ranks for r in g) == sp_ranks
    assert sorted(r for g in ring_group_ranks for r in g) == sp_ranks

    # Every rank's (ulysses_group, ring_group) pair must be unique -- i.e.
    # the two groupings together form a proper 2D coordinate for each rank.
    rank_to_ulysses_group = {r: i for i, g in enumerate(ulysses_group_ranks) for r in g}
    rank_to_ring_group = {r: i for i, g in enumerate(ring_group_ranks) for r in g}
    coords = {(rank_to_ulysses_group[r], rank_to_ring_group[r]) for r in sp_ranks}
    assert len(coords) == len(sp_ranks)
