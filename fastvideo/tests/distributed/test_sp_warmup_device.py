# SPDX-License-Identifier: Apache-2.0
"""The SP communication warmup must not assume the rank runs on CUDA.

``warmup_sequence_parallel_communication()`` builds throwaway tensors and pushes them
through the sequence-parallel collectives, so those tensors belong on whatever
accelerator the rank was placed on. Hardcoding CUDA there makes the warmup fail on
every other accelerator (Ascend NPU, MPS, ...).

These tests run without an accelerator: the platform and the SP group are replaced by
recording stand-ins, and the assertion is on the device the warmup tensors were built
with.
"""

from __future__ import annotations

import torch

from fastvideo.distributed import communication_op


class _RecordingSpGroup:
    """Stand-in for the SP group that remembers the tensors handed to the collectives."""

    def __init__(self) -> None:
        self.tensors: list[torch.Tensor] = []

    def all_to_all_4D(self, tensor: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1) -> torch.Tensor:
        self.tensors.append(tensor)
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self.tensors.append(tensor)
        return tensor


class _RecordingDeviceModule:
    """torch.cuda / torch.npu stand-in: records the devices it was asked to synchronize."""

    def __init__(self) -> None:
        self.synchronized: list = []

    def synchronize(self, device=None) -> None:
        self.synchronized.append(device)


class _RecordingPlatform:
    """Minimal stand-in for fastvideo.platforms.current_platform."""

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
    platform = _RecordingPlatform(device_type="cpu")
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
