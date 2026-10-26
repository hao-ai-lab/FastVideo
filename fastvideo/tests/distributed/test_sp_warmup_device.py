# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from fastvideo.distributed import communication_op


class _RecordingSpGroup:
    def __init__(self) -> None:
        self.tensors: list[torch.Tensor] = []

    def all_to_all_4D(self, tensor: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1) -> torch.Tensor:
        self.tensors.append(tensor)
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self.tensors.append(tensor)
        return tensor


class _RecordingDeviceModule:
    def __init__(self) -> None:
        self.synchronized: list = []

    def synchronize(self, device=None) -> None:
        self.synchronized.append(device)


class _RecordingPlatform:
    def __init__(self, device_type: str) -> None:
        self.device_type = device_type
        self.device_module = _RecordingDeviceModule()

    def get_torch_device(self) -> _RecordingDeviceModule:
        return self.device_module


def _patch_warmup_inputs(monkeypatch, platform: _RecordingPlatform, group: _RecordingSpGroup) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform", platform)
    monkeypatch.setattr(communication_op, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(communication_op, "get_sp_world_size", lambda: 2)
    monkeypatch.setattr(communication_op, "get_sp_group", lambda: group)
    monkeypatch.setattr(communication_op, "_sp_warmup_done", False)


def test_warmup_uses_the_active_platform_device(monkeypatch) -> None:
    platform = _RecordingPlatform(device_type="cpu")
    group = _RecordingSpGroup()
    _patch_warmup_inputs(monkeypatch, platform, group)

    communication_op.warmup_sequence_parallel_communication()

    assert group.tensors, "the warmup must exercise the SP collectives"
    assert {tensor.device.type for tensor in group.tensors} == {"cpu"}
    assert platform.device_module.synchronized == [torch.device("cpu")]


def test_warmup_keeps_an_explicit_device(monkeypatch) -> None:
    platform = _RecordingPlatform(device_type="cuda")
    group = _RecordingSpGroup()
    _patch_warmup_inputs(monkeypatch, platform, group)

    communication_op.warmup_sequence_parallel_communication(torch.device("cpu"))

    assert {tensor.device.type for tensor in group.tensors} == {"cpu"}
    assert platform.device_module.synchronized == [torch.device("cpu")]


def test_warmup_is_skipped_without_sequence_parallelism(monkeypatch) -> None:
    platform = _RecordingPlatform(device_type="cpu")
    group = _RecordingSpGroup()
    _patch_warmup_inputs(monkeypatch, platform, group)
    monkeypatch.setattr(communication_op, "get_sp_world_size", lambda: 1)

    communication_op.warmup_sequence_parallel_communication()

    assert group.tensors == []
    assert platform.device_module.synchronized == []
