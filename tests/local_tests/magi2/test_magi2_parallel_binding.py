# SPDX-License-Identifier: Apache-2.0
"""FastVideo process-group binding coverage for MAGI-2."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastvideo.models.dits.magi2_runtime import fastvideo_parallel
from fastvideo.models.dits.magi2_runtime import psm as psm_manager


def test_fastvideo_groups_are_reused_for_context_and_expert_parallelism(monkeypatch) -> None:
    """Bind one eight-rank sequence group to both MAGI-2 parallel axes."""
    sequence_process_group = object()
    data_process_group = object()
    sequence_group = SimpleNamespace(
        world_size=8,
        device_group=sequence_process_group,
        ranks=list(range(8)),
    )
    data_group = SimpleNamespace(
        world_size=1,
        device_group=data_process_group,
        ranks=[3],
    )
    monkeypatch.setattr(fastvideo_parallel, "get_sp_group", lambda: sequence_group)
    monkeypatch.setattr(fastvideo_parallel, "get_dp_group", lambda: data_group)
    monkeypatch.setattr(
        psm_manager,
        "get_world_size",
        lambda dim="": 8 if dim in {"cp", "ep"} else 1,
    )

    recorded: dict = {}
    monkeypatch.setattr(
        fastvideo_parallel,
        "bind_process_groups",
        lambda **kwargs: recorded.update(kwargs),
    )
    fastvideo_parallel.bind_fastvideo_parallel_state()

    assert recorded["cp_group"] is sequence_process_group
    assert recorded["ep_group"] is sequence_process_group
    assert recorded["dp_group"] is data_process_group
    assert recorded["cp_ranks"] == list(range(8))


def test_fastvideo_binding_requires_the_published_parallel_size(monkeypatch) -> None:
    """Reject sequence groups that cannot match the published eight-rank layout."""
    monkeypatch.setattr(
        fastvideo_parallel,
        "get_sp_group",
        lambda: SimpleNamespace(world_size=4),
    )
    monkeypatch.setattr(
        fastvideo_parallel,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    with pytest.raises(ValueError, match="sp_size=8"):
        fastvideo_parallel.bind_fastvideo_parallel_state()
