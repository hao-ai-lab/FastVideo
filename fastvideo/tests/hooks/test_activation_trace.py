# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import torch
from torch import nn

import fastvideo.envs as envs
from fastvideo.hooks.activation_trace import (
    attach_activation_trace,
    detach_activation_trace,
    trace_step,
)
from fastvideo.hooks.hooks import ModuleHookManager


class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        self.other = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.other(self.block(x))


class TupleLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.proj(x)
        return out, out + 1


class TupleOutputModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tuple = TupleLayer()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tuple(x)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_attach_activation_trace_off_returns_none(env_overrides) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_ACTIVATIONS.override(None))
    model = ToyModel()

    manager = attach_activation_trace(model)

    assert manager is None
    assert len(model._forward_hooks) == 0
    assert ModuleHookManager.get_from(model.block[0]) is None


def test_attach_activation_trace_on_respects_layer_filter(
    env_overrides,
    tmp_path,
) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_ACTIVATIONS.override(True))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_LAYERS.override(r"block\.0.*"))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_OUTPUT.override(str(tmp_path / "trace.jsonl")))
    model = ToyModel()

    manager = attach_activation_trace(model)

    try:
        assert manager is not None
        assert ModuleHookManager.get_from(model.block[0]) is not None
        assert ModuleHookManager.get_from(model.block[1]) is None
        assert ModuleHookManager.get_from(model.other) is None
    finally:
        detach_activation_trace(manager)


def test_activation_trace_writes_configured_stats(env_overrides, tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_ACTIVATIONS.override(True))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_LAYERS.override(r"block\.0"))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_STATS.override("abs_mean,sum,shape,dtype"))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_OUTPUT.override(str(path)))
    model = ToyModel()
    manager = attach_activation_trace(model)

    try:
        with trace_step(3):
            model(torch.ones(1, 2))
    finally:
        detach_activation_trace(manager)

    records = _read_jsonl(path)
    assert len(records) == 1
    record = records[0]
    assert record["module"] == "block.0"
    assert record["tensor"] == "out"
    assert record["step"] == 3
    assert {"abs_mean", "sum", "shape", "dtype"}.issubset(record)
    assert record["shape"] == [1, 2]
    assert record["dtype"] == "torch.float32"


def test_activation_trace_step_filter(env_overrides, tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_ACTIVATIONS.override(True))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_LAYERS.override(r"block\.0"))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_OUTPUT.override(str(path)))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_STEPS.override("0,2"))
    model = ToyModel()
    manager = attach_activation_trace(model)

    try:
        for step_idx in range(4):
            with trace_step(step_idx):
                model(torch.ones(1, 2))
    finally:
        detach_activation_trace(manager)

    assert [record["step"] for record in _read_jsonl(path)] == [0, 2]


def test_activation_trace_flattens_tuple_outputs(env_overrides, tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_ACTIVATIONS.override(True))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_LAYERS.override("tuple$"))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_OUTPUT.override(str(path)))
    model = TupleOutputModel()
    manager = attach_activation_trace(model)

    try:
        model(torch.ones(1, 2))
    finally:
        detach_activation_trace(manager)

    records = _read_jsonl(path)
    assert [record["tensor"] for record in records] == ["out[0]", "out[1]"]


def test_detach_activation_trace_removes_hooks(env_overrides, tmp_path) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_ACTIVATIONS.override(True))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_LAYERS.override(r"block\.0"))
    env_overrides.enter_context(envs.FASTVIDEO_TRACE_OUTPUT.override(str(tmp_path / "trace.jsonl")))
    model = ToyModel()
    manager = attach_activation_trace(model)

    assert ModuleHookManager.get_from(model.block[0]) is not None

    detach_activation_trace(manager)

    assert ModuleHookManager.get_from(model.block[0]) is None
