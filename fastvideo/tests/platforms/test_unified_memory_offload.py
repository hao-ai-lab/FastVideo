# SPDX-License-Identifier: Apache-2.0
"""CPU tests for worker-local offload policy on unified-memory devices."""
from __future__ import annotations

import dataclasses
from unittest.mock import Mock

import pytest

from fastvideo.api.device_policy import (UNIFIED_MEMORY_OFFLOAD_PATHS, disable_offload_on_unified_memory,
                                         finalize_device_offload_policy)
from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.api.schema import OffloadConfig
from fastvideo.api.training_schema import resolve_training_config
from fastvideo.tests.api.config_snapshot import isolated_environment

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
UNIFIED_MEMORY_OFFLOAD_FLAGS = tuple(UNIFIED_MEMORY_OFFLOAD_PATHS)


def _offload_field(flag: str) -> str:
    """The ``engine.offload`` field name of a device-policy offload flag."""
    return UNIFIED_MEMORY_OFFLOAD_PATHS[flag].rsplit(".", 1)[1]


def _resolve(engine: dict | None = None):
    """A resolved inference config for a registered model with the given ``engine`` section."""
    with isolated_environment():
        return resolve_inference_config({"model_path": WAN_T2V, "engine": engine or {}})


def _with_offloads(*enabled_flags: str, **engine):
    """A resolved inference config whose host offload modes are on exactly for ``enabled_flags``."""
    offload = {_offload_field(flag): flag in enabled_flags for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS}
    return _resolve({"offload": offload, **engine})


def _enabled_flags(resolved_config) -> list[str]:
    """The host offload modes that are on in ``resolved_config``."""
    return [
        flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(resolved_config.engine.offload, _offload_field(flag))
    ]


@pytest.fixture
def as_unified_cuda(monkeypatch):
    probe = Mock(return_value=True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    return probe


def test_resolving_an_inference_config_defers_device_policy(monkeypatch) -> None:
    probe = Mock(side_effect=AssertionError("device probe ran in the driver"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)

    resolved_config = _resolve({"use_fsdp_inference": True})

    assert resolved_config.engine.use_fsdp_inference is True
    assert resolved_config.engine.offload.dit_layerwise is True
    assert resolved_config.engine.offload.dit is True
    probe.assert_not_called()


def test_training_resolution_retains_offload_conflict_normalization(monkeypatch) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    with isolated_environment():
        resolved_config = resolve_training_config({
            "model_path": WAN_T2V,
            "engine": {
                "use_fsdp_inference": True,
                "offload": {
                    "dit": True,
                    "dit_layerwise": True
                },
                "parallelism": {
                    "sp_size": 1,
                    "hsdp_shard_dim": 1
                },
            },
        })

    assert resolved_config.engine.offload.dit_layerwise is True
    assert resolved_config.engine.offload.dit is False
    assert resolved_config.engine.use_fsdp_inference is False


def test_policy_list_covers_every_host_offload_field() -> None:
    declared = {field.name for field in dataclasses.fields(OffloadConfig)} - {"pin_cpu_memory", "lazy_module_load"}

    assert declared == {_offload_field(flag) for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS}


def test_unified_device_disables_every_offload_flag(as_unified_cuda) -> None:
    resolved_config = _resolve()

    decided = finalize_device_offload_policy(resolved_config, device_id=6)

    as_unified_cuda.assert_called_once_with(6)
    assert not _enabled_flags(decided)
    assert _enabled_flags(resolved_config) == list(UNIFIED_MEMORY_OFFLOAD_FLAGS)


@pytest.mark.parametrize("flag", UNIFIED_MEMORY_OFFLOAD_FLAGS)
def test_each_offload_flag_is_independently_disabled(as_unified_cuda, flag: str) -> None:
    resolved_config = _with_offloads(flag)
    assert _enabled_flags(resolved_config) == [flag]

    decided = disable_offload_on_unified_memory(resolved_config, device_id=2)

    assert not _enabled_flags(decided)
    assert decided.override_log == (("device_policy:unified_memory", {UNIFIED_MEMORY_OFFLOAD_PATHS[flag]: False}), )


def test_discrete_device_classification_preserves_offload_requests(monkeypatch) -> None:
    probe = Mock(return_value=False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    resolved_config = _resolve()

    assert disable_offload_on_unified_memory(resolved_config, device_id=3) is resolved_config

    probe.assert_called_once_with(3)
    assert _enabled_flags(resolved_config) == list(UNIFIED_MEMORY_OFFLOAD_FLAGS)


def test_discrete_device_finalization_retains_layerwise_precedence(monkeypatch) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    resolved_config = _resolve({"use_fsdp_inference": True})

    decided = finalize_device_offload_policy(resolved_config, device_id=3)

    offload = decided.engine.offload
    assert offload.dit_layerwise is True
    assert offload.dit is False
    assert decided.engine.use_fsdp_inference is False
    assert offload.text_encoder is True
    assert offload.image_encoder is True
    assert offload.vae is True
    assert offload.lazy_module_load is False


def test_workers_classify_their_own_device(monkeypatch) -> None:
    seen_device_ids = []

    def has_unified_memory(device_id):
        seen_device_ids.append(device_id)
        return device_id == 1

    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", has_unified_memory)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    resolved_config = _resolve()

    device_zero = finalize_device_offload_policy(resolved_config, device_id=0)
    device_one = finalize_device_offload_policy(resolved_config, device_id=1)

    assert seen_device_ids == [0, 1]
    assert device_zero.engine.offload.dit_layerwise is True
    assert device_zero.engine.offload.text_encoder is True
    assert not _enabled_flags(device_one)


def test_policy_applied_to_its_own_result_changes_nothing(as_unified_cuda) -> None:
    decided = finalize_device_offload_policy(_resolve(), device_id=1)

    again = finalize_device_offload_policy(decided, device_id=1)

    assert again.override_log == decided.override_log
    assert again.to_dict() == decided.to_dict()


def test_mps_clears_offload_and_keeps_its_fsdp_rule(monkeypatch) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "mps")
    resolved_config = _resolve({"use_fsdp_inference": True})

    decided = finalize_device_offload_policy(resolved_config)

    assert decided.engine.use_fsdp_inference is False
    assert not _enabled_flags(decided)


def test_cuda_unified_memory_preserves_realistic_fsdp_request(as_unified_cuda) -> None:
    resolved_config = _resolve({"use_fsdp_inference": True})
    assert resolved_config.engine.offload.dit_layerwise is True
    assert resolved_config.engine.use_fsdp_inference is True

    decided = finalize_device_offload_policy(resolved_config)

    assert decided.engine.use_fsdp_inference is True
    assert not _enabled_flags(decided)


def test_pin_cpu_memory_is_not_a_host_offload_mode(as_unified_cuda) -> None:
    resolved_config = _resolve({"offload": {"pin_cpu_memory": True}})

    decided = finalize_device_offload_policy(resolved_config)

    assert "pin_cpu_memory" not in UNIFIED_MEMORY_OFFLOAD_FLAGS
    assert decided.engine.offload.pin_cpu_memory is True


def test_already_disabled_flags_stay_disabled(as_unified_cuda) -> None:
    decided = finalize_device_offload_policy(_with_offloads())

    assert not _enabled_flags(decided)


@pytest.mark.parametrize("name_error", [NotImplementedError, ValueError, RuntimeError])
def test_platform_without_device_name_uses_generic_name(monkeypatch, name_error: type[Exception]) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)

    def unsupported_name(device_id):
        raise name_error("device name unavailable")

    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", unsupported_name)
    resolved_config = _with_offloads("text_encoder_cpu_offload")

    decided = disable_offload_on_unified_memory(resolved_config, device_id=0)

    assert decided.engine.offload.text_encoder is False


def test_unified_device_auto_enables_lazy_module_load(as_unified_cuda) -> None:
    resolved_config = _resolve()

    assert resolved_config.engine.offload.lazy_module_load is None
    assert finalize_device_offload_policy(resolved_config, device_id=6).engine.offload.lazy_module_load is True


def test_explicit_false_lazy_module_load_stays_off_on_unified(as_unified_cuda) -> None:
    resolved_config = _resolve({"offload": {"lazy_module_load": False}})

    decided = finalize_device_offload_policy(resolved_config, device_id=6)

    assert decided.engine.offload.lazy_module_load is False
