# SPDX-License-Identifier: Apache-2.0
"""CPU tests for worker-local offload policy on unified-memory devices."""
from __future__ import annotations

import dataclasses
from unittest.mock import Mock

import pytest

from fastvideo.fastvideo_args import ExecutionMode, UNIFIED_MEMORY_OFFLOAD_FLAGS, FastVideoArgs


def _args_with_offloads(*enabled_flags: str, **overrides) -> FastVideoArgs:
    kwargs = {flag: flag in enabled_flags for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS}
    kwargs.update(model_path="unused/for-this-test")
    kwargs.update(overrides)
    return FastVideoArgs(**kwargs)


@pytest.fixture
def as_unified_cuda(monkeypatch):
    probe = Mock(return_value=True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    return probe


def test_constructing_inference_args_defers_device_policy(monkeypatch) -> None:
    probe = Mock(side_effect=AssertionError("device probe ran in the driver"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)

    args = FastVideoArgs(model_path="unused/for-this-test", use_fsdp_inference=True)

    assert args.use_fsdp_inference is True
    assert args.dit_layerwise_offload is True
    assert args.dit_cpu_offload is True
    probe.assert_not_called()


def test_training_construction_retains_offload_conflict_normalization(monkeypatch) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    args = FastVideoArgs(
        model_path="unused/for-this-test",
        mode=ExecutionMode.FINETUNING,
        inference_mode=False,
        sp_size=1,
        hsdp_shard_dim=1,
        use_fsdp_inference=True,
    )

    assert args.dit_layerwise_offload is True
    assert args.dit_cpu_offload is False
    assert args.use_fsdp_inference is False


def test_policy_list_covers_every_declared_offload_flag() -> None:
    declared = {field.name for field in dataclasses.fields(FastVideoArgs) if field.name.endswith("_offload")}

    assert declared == set(UNIFIED_MEMORY_OFFLOAD_FLAGS)


def test_unified_device_disables_every_offload_flag(as_unified_cuda) -> None:
    args = FastVideoArgs(model_path="unused/for-this-test")

    assert args.finalize_device_offload_policy(device_id=6) is True

    as_unified_cuda.assert_called_once_with(6)
    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]


@pytest.mark.parametrize("flag", UNIFIED_MEMORY_OFFLOAD_FLAGS)
def test_each_offload_flag_is_independently_disabled(as_unified_cuda, flag: str) -> None:
    args = _args_with_offloads(flag)
    assert getattr(args, flag) is True

    args.disable_offload_on_unified_memory(device_id=2)

    assert getattr(args, flag) is False


def test_discrete_device_classification_preserves_offload_requests(monkeypatch) -> None:
    probe = Mock(return_value=False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    args = FastVideoArgs(model_path="unused/for-this-test")

    assert args.disable_offload_on_unified_memory(device_id=3) is False

    probe.assert_called_once_with(3)
    assert all(getattr(args, flag) for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS)


def test_discrete_device_finalization_retains_layerwise_precedence(monkeypatch) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    args = FastVideoArgs(model_path="unused/for-this-test", use_fsdp_inference=True)

    args.finalize_device_offload_policy(device_id=3)

    assert args.dit_layerwise_offload is True
    assert args.dit_cpu_offload is False
    assert args.use_fsdp_inference is False
    assert args.text_encoder_cpu_offload is True
    assert args.image_encoder_cpu_offload is True
    assert args.vae_cpu_offload is True


def test_workers_classify_their_own_device(monkeypatch) -> None:
    seen_device_ids = []

    def has_unified_memory(device_id):
        seen_device_ids.append(device_id)
        return device_id == 1

    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", has_unified_memory)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    device_zero_args = FastVideoArgs(model_path="unused/for-this-test")
    device_one_args = FastVideoArgs(model_path="unused/for-this-test")

    device_zero_args.finalize_device_offload_policy(device_id=0)
    device_one_args.finalize_device_offload_policy(device_id=1)

    assert seen_device_ids == [0, 1]
    assert device_zero_args.dit_layerwise_offload is True
    assert device_zero_args.text_encoder_cpu_offload is True
    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(device_one_args, flag)]


def test_repeated_policy_on_same_device_reuses_classification(as_unified_cuda) -> None:
    args = FastVideoArgs(model_path="unused/for-this-test")

    args.finalize_device_offload_policy(device_id=1)
    args.finalize_device_offload_policy(device_id=1)

    as_unified_cuda.assert_called_once_with(1)


def test_mps_clears_offload_and_keeps_its_fsdp_rule(monkeypatch) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "mps")
    args = FastVideoArgs(model_path="unused/for-this-test", use_fsdp_inference=True)

    args.finalize_device_offload_policy()

    assert args.use_fsdp_inference is False
    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]


def test_cuda_unified_memory_preserves_realistic_fsdp_request(as_unified_cuda) -> None:
    args = FastVideoArgs(model_path="unused/for-this-test", use_fsdp_inference=True)
    assert args.dit_layerwise_offload is True
    assert args.use_fsdp_inference is True

    args.finalize_device_offload_policy()

    assert args.use_fsdp_inference is True
    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]


def test_pin_cpu_memory_is_not_a_host_offload_mode(as_unified_cuda) -> None:
    args = FastVideoArgs(model_path="unused/for-this-test", pin_cpu_memory=True)

    args.finalize_device_offload_policy()

    assert "pin_cpu_memory" not in UNIFIED_MEMORY_OFFLOAD_FLAGS
    assert args.pin_cpu_memory is True


def test_already_disabled_flags_stay_disabled(as_unified_cuda) -> None:
    args = _args_with_offloads()

    args.finalize_device_offload_policy()

    assert not [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]


@pytest.mark.parametrize("name_error", [NotImplementedError, ValueError, RuntimeError])
def test_platform_without_device_name_uses_generic_name(monkeypatch, name_error: type[Exception]) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)

    def unsupported_name(device_id):
        raise name_error("device name unavailable")

    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", unsupported_name)
    args = _args_with_offloads("text_encoder_cpu_offload")

    assert args.disable_offload_on_unified_memory() is True
    assert args.text_encoder_cpu_offload is False
