# SPDX-License-Identifier: Apache-2.0
"""Regression tests for encoder placement: the loader reads the offload decision that resolution made."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from fastvideo.api import device_policy
from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.models.loader.component_loader import ImageEncoderLoader, TextEncoderLoader
from fastvideo.tests.api.config_snapshot import isolated_environment

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class _PassthroughEncoder(nn.Module):
    supports_hf_from_pretrained = True
    loaded_device: torch.device | None = None

    @classmethod
    def from_pretrained_local(cls, model_path, model_config, *, dtype, device):
        del model_path, model_config, dtype
        cls.loaded_device = torch.device(device)
        return cls()


def _model_config():
    return SimpleNamespace(architectures=["PassthroughEncoder"], _fsdp_shard_conditions=[], quant_config=None)


def _resolved_config(monkeypatch, *, unified: bool, **offload):
    """A resolved inference config for a registered model, resolved on a unified or discrete device."""
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: unified)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    with isolated_environment(), patch.object(device_policy, "APPLY_DEVICE_POLICY", True):
        return resolve_inference_config({"model_path": WAN_T2V, "engine": {"offload": offload}})


@pytest.fixture
def passthrough_registry(monkeypatch):
    monkeypatch.setattr(
        "fastvideo.models.loader.component_loader.ModelRegistry.resolve_model_cls",
        lambda architectures: (_PassthroughEncoder, None),
    )


@pytest.mark.parametrize("requested_target", [torch.device("cuda:5"), torch.device("cpu")])
def test_unified_memory_decision_keeps_the_encoder_on_the_worker_device(monkeypatch, tmp_path, passthrough_registry,
                                                                        requested_target) -> None:
    resolved_config = _resolved_config(monkeypatch, unified=True, text_encoder=True)
    assert resolved_config.engine.offload.text_encoder is False
    probe = Mock(side_effect=AssertionError("device probe ran in the loader"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr("fastvideo.models.loader.component_loader.get_local_torch_device",
                        lambda: torch.device("cuda:5"))

    model = TextEncoderLoader().load_model(str(tmp_path), _model_config(), requested_target, resolved_config)

    assert isinstance(model, _PassthroughEncoder)
    # cpu_offload is off, so the loader keeps the target that the caller chose.
    assert _PassthroughEncoder.loaded_device == requested_target
    assert resolved_config.override_log == ()
    probe.assert_not_called()


def test_discrete_memory_offloads_the_encoder_to_the_host(monkeypatch, tmp_path, passthrough_registry) -> None:
    resolved_config = _resolved_config(monkeypatch, unified=False, text_encoder=True)
    assert resolved_config.engine.offload.text_encoder is True
    monkeypatch.setattr("fastvideo.models.loader.component_loader.get_local_torch_device",
                        lambda: torch.device("cuda:2"))

    TextEncoderLoader().load_model(str(tmp_path), _model_config(), torch.device("cuda:2"), resolved_config)

    assert _PassthroughEncoder.loaded_device == torch.device("cpu")


def test_discrete_memory_preserves_explicit_cpu_target(monkeypatch, tmp_path, passthrough_registry) -> None:
    resolved_config = _resolved_config(monkeypatch, unified=False, text_encoder=False)
    monkeypatch.setattr("fastvideo.models.loader.component_loader.get_local_torch_device",
                        lambda: torch.device("cuda:2"))

    TextEncoderLoader().load_model(
        str(tmp_path),
        _model_config(),
        torch.device("cuda:2"),
        resolved_config,
        cpu_offload=True,
    )

    assert _PassthroughEncoder.loaded_device == torch.device("cpu")


def test_image_encoder_reads_its_own_offload_decision(monkeypatch, tmp_path, passthrough_registry) -> None:
    resolved_config = _resolved_config(monkeypatch, unified=True, text_encoder=True, image_encoder=True)
    assert resolved_config.engine.offload.image_encoder is False
    monkeypatch.setattr("fastvideo.models.loader.component_loader.get_local_torch_device",
                        lambda: torch.device("cuda:4"))

    ImageEncoderLoader().load_model(
        str(tmp_path),
        _model_config(),
        torch.device("cuda:4"),
        resolved_config,
        cpu_offload=resolved_config.engine.offload.image_encoder,
        offload_flag="image_encoder_cpu_offload",
    )

    assert _PassthroughEncoder.loaded_device == torch.device("cuda:4")
    assert resolved_config.override_log == ()
