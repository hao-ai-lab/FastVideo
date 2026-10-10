# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``build_usp_topology``'s ring/ulysses decision logic.

The two degenerate cases (``ring_size == 1`` and ``ulysses_size == 1``) never
call ``init_model_parallel_group`` -- they alias ``sp_group`` directly and
only ever read its ``world_size`` -- so they are tested here with a
duck-typed fake ``sp_group`` and no real process group / torch.distributed
initialization, the same way ``test_usp_group_layout.py`` tests
``_build_usp_subgroup_ranks`` without one.

The hybrid case (``ring_size > 1 and ulysses_size > 1``) does create real
subgroups via ``init_model_parallel_group``, which requires a live
distributed backend; that path is covered end-to-end by the multi-GPU tests
in ``fastvideo/tests/distributed/test_ring_attention.py`` instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastvideo.distributed.usp_topology import build_usp_topology


def _fake_sp_group(world_size: int) -> SimpleNamespace:
    """A minimal stand-in for ``GroupCoordinator``: only ``.world_size`` is
    read on the degenerate-case paths under test."""
    return SimpleNamespace(world_size=world_size)


def test_ring_size_one_is_pure_ulysses() -> None:
    """``ring_size=1`` degenerates to pure Ulysses: the Ulysses group is
    ``sp_group`` itself and no Ring group is created."""
    sp_group = _fake_sp_group(world_size=4)

    topology = build_usp_topology(
        sp_group=sp_group,
        sp_group_ranks=[[0, 1, 2, 3]],
        ring_size=1,
        local_rank=0,
        backend="gloo",
    )

    assert topology.ring_size == 1
    assert topology.ulysses_size == 4
    assert topology.ring_group is None
    assert topology.ulysses_group is sp_group


def test_ulysses_size_one_is_pure_ring() -> None:
    """``ring_size == sp_size`` degenerates to pure Ring: the Ring group is
    ``sp_group`` itself and no Ulysses group is created."""
    sp_group = _fake_sp_group(world_size=4)

    topology = build_usp_topology(
        sp_group=sp_group,
        sp_group_ranks=[[0, 1, 2, 3]],
        ring_size=4,
        local_rank=0,
        backend="gloo",
    )

    assert topology.ring_size == 4
    assert topology.ulysses_size == 1
    assert topology.ring_group is sp_group
    assert topology.ulysses_group is None


def test_ring_size_must_divide_sp_size() -> None:
    """``sp_size`` not divisible by ``ring_size`` must fail fast rather than
    silently truncate into an incomplete mesh."""
    sp_group = _fake_sp_group(world_size=4)

    with pytest.raises(AssertionError, match="must be divisible by"):
        build_usp_topology(
            sp_group=sp_group,
            sp_group_ranks=[[0, 1, 2, 3]],
            ring_size=3,
            local_rank=0,
            backend="gloo",
        )
