# SPDX-License-Identifier: Apache-2.0
"""CPU tests for worker-local offload policy on unified-memory devices."""
from __future__ import annotations

import dataclasses
from unittest.mock import Mock

import pytest

from fastvideo.fastvideo_args import UNIFIED_MEMORY_OFFLOAD_FLAGS, FastVideoArgs


def _args(**overrides) -> FastVideoArgs:
    kwargs = {flag: False for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS}
    kwargs.update(model_path="unused/for-this-test")
    kwargs.update(overrides)
    return FastVideoArgs(**kwargs)


def _enable_every_offload(args: FastVideoArgs) -> None:
    for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS:
        setattr(args, flag, True)


@pytest.fixture
def as_unified_device(monkeypatch):
    probe = Mock(return_value=True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    return probe


def test_constructing_args_does_not_probe_runtime_device_properties(monkeypatch) -> None:
    probe = Mock(side_effect=AssertionError("device probe ran in the driver"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)

    args = _args(text_encoder_cpu_offload=True)

    assert args.text_encoder_cpu_offload is True
    probe.assert_not_called()


def test_policy_list_covers_every_declared_offload_flag() -> None:
    declared = {field.name for field in dataclasses.fields(FastVideoArgs) if field.name.endswith("_offload")}

    assert declared == set(UNIFIED_MEMORY_OFFLOAD_FLAGS)


def test_unified_device_disables_every_offload_flag(as_unified_device) -> None:
    args = _args()
    _enable_every_offload(args)

    assert args.disable_offload_on_unified_memory(device_id=6) is True

    as_unified_device.assert_called_once_with(6)
    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]


@pytest.mark.parametrize("flag", UNIFIED_MEMORY_OFFLOAD_FLAGS)
def test_each_offload_flag_is_independently_disabled(as_unified_device, flag: str) -> None:
    # Every other flag starts false so dit_cpu_offload cannot pass vacuously
    # through the constructor-time precedence rule for layerwise offload.
    args = _args(**{flag: True})
    assert getattr(args, flag) is True

    args.disable_offload_on_unified_memory(device_id=2)

    assert getattr(args, flag) is False


def test_discrete_device_preserves_every_offload_flag(monkeypatch) -> None:
    probe = Mock(return_value=False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    args = _args()
    _enable_every_offload(args)

    assert args.disable_offload_on_unified_memory(device_id=3) is False

    probe.assert_called_once_with(3)
    assert all(getattr(args, flag) for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS)


def test_workers_classify_their_own_device(monkeypatch) -> None:
    seen_device_ids = []

    def has_unified_memory(device_id):
        seen_device_ids.append(device_id)
        return device_id == 1

    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", has_unified_memory)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    device_zero_args = _args()
    device_one_args = _args()
    _enable_every_offload(device_zero_args)
    _enable_every_offload(device_one_args)

    device_zero_args.disable_offload_on_unified_memory(device_id=0)
    device_one_args.disable_offload_on_unified_memory(device_id=1)

    assert seen_device_ids == [0, 1]
    assert all(getattr(device_zero_args, flag) for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS)
    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(device_one_args, flag)]


def test_mps_clears_offload_and_keeps_its_fsdp_rule(monkeypatch) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: True)
    args = _args(use_fsdp_inference=True)
    _enable_every_offload(args)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "mps")

    args.disable_offload_on_unified_memory()

    assert args.use_fsdp_inference is False
    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]


def test_cuda_unified_memory_does_not_change_fsdp_inference(as_unified_device) -> None:
    args = _args(use_fsdp_inference=True)
    _enable_every_offload(args)

    args.disable_offload_on_unified_memory()

    assert args.use_fsdp_inference is True
    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]


def test_already_disabled_flags_stay_disabled(as_unified_device) -> None:
    args = _args()

    args.disable_offload_on_unified_memory()

    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]


@pytest.mark.parametrize("name_error", [NotImplementedError, ValueError, RuntimeError])
def test_platform_without_device_name_uses_generic_name(monkeypatch, name_error: type[Exception]) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)

    def unsupported_name(device_id):
        raise name_error("device name unavailable")

    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", unsupported_name)
    args = _args(text_encoder_cpu_offload=True)

    assert args.disable_offload_on_unified_memory() is True
    assert args.text_encoder_cpu_offload is False
