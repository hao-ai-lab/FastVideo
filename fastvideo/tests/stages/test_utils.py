# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

import fastvideo.pipelines.stages.utils as stage_utils


class _Parameter:

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)


class _Module:

    def __init__(self, *parameters: Any) -> None:
        self._parameters = list(parameters)
        self.moves: list[torch.device] = []

    def parameters(self):
        return iter(self._parameters)

    def to(self, device: str | torch.device):
        resolved = torch.device(device)
        self.moves.append(resolved)
        for parameter in self._parameters:
            parameter.device = resolved
        return self


def test_module_device_uses_plain_parameter_device() -> None:
    module = _Module(_Parameter("cuda:1"))

    assert stage_utils.module_device(module) == torch.device("cuda:1")


def test_module_device_uses_configured_device_for_parameterless_module() -> None:
    module = _Module()
    module._fastvideo_input_device = torch.device("cuda:2")

    assert stage_utils.module_device(module) == torch.device("cuda:2")
    assert stage_utils.module_device(_Module()) == torch.device("cpu")
    assert stage_utils.module_device(_Module(), fallback=torch.device("meta")) == torch.device("meta")


def test_move_module_moves_plain_module_to_local_device(monkeypatch) -> None:
    module = _Module(_Parameter("cpu"))
    monkeypatch.setattr(stage_utils, "get_local_torch_device", lambda: torch.device("cuda:0"))

    result, input_device, moved = stage_utils.move_module_to_local_device(module)

    assert result is module
    assert input_device == torch.device("cuda:0")
    assert moved
    assert module.moves == [torch.device("cuda:0")]


def test_move_module_finds_dtensor_after_plain_parameter(monkeypatch) -> None:

    class _DTensorParameter(_Parameter):
        pass

    module = _Module(_Parameter("cpu"), _DTensorParameter("cpu"))
    monkeypatch.setattr(stage_utils, "DTensor", _DTensorParameter)
    monkeypatch.setattr(stage_utils, "get_local_torch_device", lambda: torch.device("cuda:0"))

    assert stage_utils.module_device(module) == torch.device("cpu")
    result, input_device, moved = stage_utils.move_module_to_local_device(module)

    assert result is module
    assert input_device == torch.device("cpu")
    assert not moved
    assert module.moves == []


def test_maybe_offload_module_only_moves_when_enabled() -> None:
    module = _Module(_Parameter("cuda:0"))

    assert stage_utils.maybe_offload_module(module, enabled=False) is module
    assert module.moves == []
    assert stage_utils.maybe_offload_module(module, enabled=True) is module
    assert module.moves == [torch.device("cpu")]
