# SPDX-License-Identifier: Apache-2.0
"""CPU tests for applying unified-memory offload policy inside a worker."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from fastvideo.fastvideo_args import FastVideoArgs


def _args(**overrides) -> FastVideoArgs:
    return FastVideoArgs(model_path="unused/for-this-test", **overrides)


def test_constructing_args_does_not_probe_runtime_device_properties(monkeypatch) -> None:
    probe = Mock(side_effect=AssertionError("device probe ran in the driver"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)

    args = _args(text_encoder_cpu_offload=True)

    assert args.text_encoder_cpu_offload is True
    probe.assert_not_called()


def test_unified_device_disables_text_encoder_offload_for_selected_device(monkeypatch) -> None:
    probe = Mock(return_value=True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    args = _args(text_encoder_cpu_offload=True)

    assert args.disable_offload_on_unified_memory(device_id=6, offload_flag="text_encoder_cpu_offload") is True

    probe.assert_called_once_with(6)
    assert args.text_encoder_cpu_offload is False


def test_discrete_device_keeps_text_encoder_offload(monkeypatch) -> None:
    probe = Mock(return_value=False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    args = _args(text_encoder_cpu_offload=True)

    assert args.disable_offload_on_unified_memory(device_id=3) is False

    probe.assert_called_once_with(3)
    assert args.text_encoder_cpu_offload is True


def test_policy_does_not_claim_unlisted_component_role(monkeypatch) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    args = _args(text_encoder_cpu_offload=True, image_encoder_cpu_offload=True)

    applies_to_image_encoder = args.disable_offload_on_unified_memory(
        device_id=0,
        offload_flag="image_encoder_cpu_offload",
    )

    assert applies_to_image_encoder is False
    assert args.text_encoder_cpu_offload is False
    assert args.image_encoder_cpu_offload is True


@pytest.mark.parametrize("name_error", [NotImplementedError, ValueError, RuntimeError])
def test_platform_without_device_name_uses_generic_name(monkeypatch, name_error: type[Exception]) -> None:
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)

    def unsupported_name(device_id):
        raise name_error("device name unavailable")

    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", unsupported_name)
    args = _args(text_encoder_cpu_offload=True)

    assert args.disable_offload_on_unified_memory() is True
    assert args.text_encoder_cpu_offload is False
