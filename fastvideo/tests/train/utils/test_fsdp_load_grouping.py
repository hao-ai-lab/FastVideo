# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

from fastvideo.models.loader import fsdp_load


class _Block(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.sensitive = torch.nn.Parameter(torch.ones(1))


class _Model(torch.nn.Module):

    def __init__(self, block_count: int = 4) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(block_count)])
        self.root_weight = torch.nn.Parameter(torch.ones(1))
        self.root_sensitive = torch.nn.Parameter(torch.ones(1))


class _MixedDtypeModel(_Model):

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        return torch.float32 if name.endswith("sensitive") else default_dtype


def _is_block(name: str, module: torch.nn.Module) -> bool:
    return isinstance(module, _Block) and name.startswith("blocks.")


def _record_fully_shard(monkeypatch) -> list[tuple[Any, dict[str, Any]]]:
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.delenv("FASTVIDEO_FSDP2_AUTOWRAP", raising=False)
    monkeypatch.setattr(fsdp_load, "fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))
    return calls


def _shard(model: torch.nn.Module, *, modules_per_group: int = 1) -> None:
    fsdp_load.shard_model(
        model,
        cpu_offload=False,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        mesh=object(),
        fsdp_shard_conditions=[_is_block],
        fsdp_modules_per_group=modules_per_group,
    )


def test_default_shards_each_matching_module_in_existing_reverse_order(monkeypatch) -> None:
    model = _Model()
    calls = _record_fully_shard(monkeypatch)

    _shard(model)

    assert [module for module, _ in calls] == [*reversed(model.blocks), model]
    assert all(not isinstance(module, list) for module, _ in calls)


def test_groups_adjacent_matching_modules_and_shards_root_last(monkeypatch) -> None:
    model = _Model()
    calls = _record_fully_shard(monkeypatch)

    _shard(model, modules_per_group=2)

    assert [module for module, _ in calls] == [
        [model.blocks[0], model.blocks[1]],
        [model.blocks[2], model.blocks[3]],
        model,
    ]


def test_grouped_modules_receive_their_combined_ignored_parameters(monkeypatch) -> None:
    model = _MixedDtypeModel(block_count=3)
    calls = _record_fully_shard(monkeypatch)

    _shard(model, modules_per_group=2)

    assert calls[0][1]["ignored_params"] == {
        model.blocks[0].sensitive,
        model.blocks[1].sensitive,
    }
    assert calls[1][1]["ignored_params"] == {model.blocks[2].sensitive}
    assert calls[2][0] is model
    assert calls[2][1]["ignored_params"] == {
        *(block.sensitive for block in model.blocks),
        model.root_sensitive,
    }


@pytest.mark.parametrize("modules_per_group", [0, -1])
def test_rejects_invalid_group_size_before_sharding(monkeypatch, modules_per_group: int) -> None:
    calls = _record_fully_shard(monkeypatch)

    with pytest.raises(ValueError, match="must be at least 1"):
        _shard(_Model(), modules_per_group=modules_per_group)

    assert calls == []


def test_rejects_grouping_with_size_based_autowrap(monkeypatch) -> None:
    calls = _record_fully_shard(monkeypatch)
    monkeypatch.setenv("FASTVIDEO_FSDP2_AUTOWRAP", "1")

    with pytest.raises(ValueError, match="incompatible with FASTVIDEO_FSDP2_AUTOWRAP"):
        _shard(_Model(), modules_per_group=2)

    assert calls == []


@pytest.mark.parametrize("shared_with", [1, 2])
def test_rejects_overlapping_parameter_sets_before_sharding(monkeypatch, shared_with: int) -> None:
    model = _Model()
    model.blocks[shared_with].weight = model.blocks[0].weight
    calls = _record_fully_shard(monkeypatch)

    with pytest.raises(ValueError, match="overlapping parameter sets"):
        _shard(model, modules_per_group=2)

    assert calls == []
