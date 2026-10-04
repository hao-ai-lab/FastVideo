# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the device-policy resolution steps, which settle host offload for the local device's memory class."""
from __future__ import annotations

import dataclasses
from unittest.mock import Mock, patch

import pytest

from fastvideo.api import device_policy
from fastvideo.api.device_policy import LOCAL_DEVICE_ID, UNIFIED_MEMORY_OFFLOAD_PATHS
from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.api.schema import OffloadConfig
from fastvideo.api.training_schema import resolve_training_config
from fastvideo.tests.api.config_snapshot import isolated_environment

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
UNIFIED_MEMORY_OFFLOAD_FLAGS = tuple(UNIFIED_MEMORY_OFFLOAD_PATHS)


def _offload_field(flag: str) -> str:
    """The ``engine.offload`` field name of a device-policy offload flag."""
    return UNIFIED_MEMORY_OFFLOAD_PATHS[flag].rsplit(".", 1)[1]


def _device_policy_enabled():
    """Turn the device-policy steps on inside the test isolation, which turns them off like downloads."""
    return patch.object(device_policy, "APPLY_DEVICE_POLICY", True)


def _resolve(engine: dict | None = None):
    """A resolved inference config for a registered model with the given ``engine`` section."""
    with isolated_environment(), _device_policy_enabled():
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


def _decisions_of(resolved_config, source: str) -> list[dict]:
    """The values that the step ``source`` decided, in order."""
    return [values for decided_by, values in resolved_config.decisions if decided_by == source]


@pytest.fixture
def as_unified_cuda(monkeypatch):
    probe = Mock(return_value=True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    return probe


@pytest.fixture
def as_discrete_cuda(monkeypatch):
    probe = Mock(return_value=False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    return probe


def test_resolution_queries_the_local_device(as_unified_cuda) -> None:
    _resolve()

    assert as_unified_cuda.call_args_list
    assert {call.args for call in as_unified_cuda.call_args_list} == {(LOCAL_DEVICE_ID, )}


def test_training_resolution_settles_offload_conflicts(as_discrete_cuda) -> None:
    with isolated_environment(), _device_policy_enabled():
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
    assert resolved_config.provenance("engine.use_fsdp_inference").source == "apply_layerwise_offload_conflicts"
    assert resolved_config.engine.offload.lazy_module_load is False


def test_policy_list_covers_every_host_offload_field() -> None:
    declared = {field.name for field in dataclasses.fields(OffloadConfig)} - {"pin_cpu_memory", "lazy_module_load"}

    assert declared == {_offload_field(flag) for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS}


def test_unified_device_disables_every_offload_flag(as_unified_cuda) -> None:
    resolved_config = _resolve()

    assert not _enabled_flags(resolved_config)
    assert _decisions_of(resolved_config, "apply_unified_memory_offload_policy") == [{
        path: False
        for path in UNIFIED_MEMORY_OFFLOAD_PATHS.values()
    }]
    for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS:
        provenance = resolved_config.provenance(UNIFIED_MEMORY_OFFLOAD_PATHS[flag])
        assert (provenance.raw_value, provenance.source) == (True, "apply_unified_memory_offload_policy")


@pytest.mark.parametrize("flag", UNIFIED_MEMORY_OFFLOAD_FLAGS)
def test_each_offload_flag_is_independently_disabled(as_unified_cuda, flag: str) -> None:
    resolved_config = _with_offloads(flag)

    assert not _enabled_flags(resolved_config)
    assert _decisions_of(resolved_config, "apply_unified_memory_offload_policy") == [{
        UNIFIED_MEMORY_OFFLOAD_PATHS[flag]: False
    }]


def test_discrete_device_classification_preserves_offload_requests(as_discrete_cuda) -> None:
    resolved_config = _resolve()

    assert _enabled_flags(resolved_config) == [
        flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if flag != "dit_cpu_offload"
    ]
    assert _decisions_of(resolved_config, "apply_unified_memory_offload_policy") == []
    assert resolved_config.provenance("engine.offload.dit").source == "apply_layerwise_offload_conflicts"


def test_discrete_device_retains_layerwise_precedence(as_discrete_cuda) -> None:
    resolved_config = _resolve({"use_fsdp_inference": True})

    offload = resolved_config.engine.offload
    assert offload.dit_layerwise is True
    assert offload.dit is False
    assert resolved_config.engine.use_fsdp_inference is False
    assert offload.text_encoder is True
    assert offload.image_encoder is True
    assert offload.vae is True
    assert offload.lazy_module_load is False
    assert _decisions_of(resolved_config, "apply_layerwise_offload_conflicts") == [{
        "engine.use_fsdp_inference": False,
        "engine.offload.dit": False,
    }]


def test_mps_clears_offload_and_keeps_its_fsdp_rule(monkeypatch) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "mps")

    resolved_config = _resolve({"use_fsdp_inference": True})

    assert resolved_config.engine.use_fsdp_inference is False
    assert not _enabled_flags(resolved_config)
    assert _decisions_of(resolved_config, "apply_mps_offload_policy") == [{
        "engine.use_fsdp_inference": False,
        "engine.offload.dit_layerwise": False,
    }]


def test_cuda_unified_memory_preserves_realistic_fsdp_request(as_unified_cuda) -> None:
    resolved_config = _resolve({"use_fsdp_inference": True})

    assert resolved_config.engine.use_fsdp_inference is True
    assert not _enabled_flags(resolved_config)
    assert _decisions_of(resolved_config, "apply_layerwise_offload_conflicts") == []


def test_pin_cpu_memory_is_not_a_host_offload_mode(as_unified_cuda) -> None:
    resolved_config = _resolve({"offload": {"pin_cpu_memory": True}})

    assert "pin_cpu_memory" not in UNIFIED_MEMORY_OFFLOAD_FLAGS
    assert resolved_config.engine.offload.pin_cpu_memory is True


def test_already_disabled_flags_stay_disabled(as_unified_cuda) -> None:
    resolved_config = _with_offloads()

    assert not _enabled_flags(resolved_config)
    assert _decisions_of(resolved_config, "apply_unified_memory_offload_policy") == []


@pytest.mark.parametrize("name_error", [NotImplementedError, ValueError, RuntimeError])
def test_platform_without_device_name_uses_generic_name(monkeypatch, name_error: type[Exception]) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    def unsupported_name(device_id):
        raise name_error("device name unavailable")

    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", unsupported_name)

    resolved_config = _with_offloads("text_encoder_cpu_offload")

    assert resolved_config.engine.offload.text_encoder is False


def test_failed_device_query_counts_as_discrete(monkeypatch) -> None:

    def unavailable(device_id):
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", unavailable)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    resolved_config = _with_offloads("text_encoder_cpu_offload")

    assert resolved_config.engine.offload.text_encoder is True
    assert resolved_config.engine.offload.lazy_module_load is False


def test_unified_device_auto_enables_lazy_module_load(as_unified_cuda) -> None:
    resolved_config = _resolve()

    provenance = resolved_config.provenance("engine.offload.lazy_module_load")
    assert (provenance.raw_value, provenance.value, provenance.source) == (None, True, "fill_lazy_module_load")


def test_explicit_false_lazy_module_load_stays_off_on_unified(as_unified_cuda) -> None:
    resolved_config = _resolve({"offload": {"lazy_module_load": False}})

    assert resolved_config.engine.offload.lazy_module_load is False
    assert resolved_config.provenance("engine.offload.lazy_module_load").source == "input"


def test_disabled_device_policy_decides_nothing(as_unified_cuda) -> None:
    with isolated_environment(), patch.object(device_policy, "APPLY_DEVICE_POLICY", False):
        resolved_config = resolve_inference_config({"model_path": WAN_T2V, "engine": {"use_fsdp_inference": True}})

    assert _enabled_flags(resolved_config) == list(UNIFIED_MEMORY_OFFLOAD_FLAGS)
    assert resolved_config.engine.use_fsdp_inference is True
    assert resolved_config.engine.offload.lazy_module_load is None
    assert [source for source, _ in resolved_config.decisions if source.startswith(("apply_", "fill_lazy"))] == []
    as_unified_cuda.assert_not_called()
