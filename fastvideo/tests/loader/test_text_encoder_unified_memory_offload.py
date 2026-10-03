# SPDX-License-Identifier: Apache-2.0
"""Regression tests for encoder placement on unified memory."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

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


def _resolved_config(**offload):
    """A resolved inference config for a registered model with the given ``engine.offload`` values."""
    with isolated_environment():
        return resolve_inference_config({"model_path": WAN_T2V, "engine": {"offload": offload}})


@pytest.mark.parametrize(
    ("cpu_offload", "requested_target"),
    [
        (None, torch.device("cuda:5")),
        (None, torch.device("cpu")),
        (True, torch.device("cpu")),
    ],
)
def test_unified_memory_uses_worker_device_before_model_construction(monkeypatch, tmp_path, cpu_offload,
                                                                     requested_target) -> None:
    probe = Mock(return_value=True)
    monkeypatch.setattr("fastvideo.models.loader.component_loader.get_local_torch_device",
                        lambda: torch.device("cuda:5"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr(
        "fastvideo.models.loader.component_loader.ModelRegistry.resolve_model_cls",
        lambda architectures: (_PassthroughEncoder, None),
    )
    resolved_config = _resolved_config(text_encoder=True)

    model = TextEncoderLoader().load_model(
        str(tmp_path),
        _model_config(),
        requested_target,
        resolved_config,
        cpu_offload=cpu_offload,
    )

    assert isinstance(model, _PassthroughEncoder)
    assert _PassthroughEncoder.loaded_device == torch.device("cuda:5")
    # The loader only queries the policy; the worker records the offload decisions on the config.
    assert resolved_config.engine.offload.text_encoder is True
    assert resolved_config.override_log == ()
    probe.assert_called_once_with(5)


def test_discrete_memory_preserves_explicit_cpu_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("fastvideo.models.loader.component_loader.get_local_torch_device",
                        lambda: torch.device("cuda:2"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr(
        "fastvideo.models.loader.component_loader.ModelRegistry.resolve_model_cls",
        lambda architectures: (_PassthroughEncoder, None),
    )
    resolved_config = _resolved_config(text_encoder=False)

    TextEncoderLoader().load_model(
        str(tmp_path),
        _model_config(),
        torch.device("cuda:2"),
        resolved_config,
        cpu_offload=True,
    )

    assert _PassthroughEncoder.loaded_device == torch.device("cpu")


def test_image_encoder_explicit_offload_resets_cpu_target(monkeypatch, tmp_path) -> None:
    probe = Mock(return_value=True)
    monkeypatch.setattr("fastvideo.models.loader.component_loader.get_local_torch_device",
                        lambda: torch.device("cuda:4"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr(
        "fastvideo.models.loader.component_loader.ModelRegistry.resolve_model_cls",
        lambda architectures: (_PassthroughEncoder, None),
    )
    resolved_config = _resolved_config(text_encoder=True, image_encoder=True)

    ImageEncoderLoader().load_model(
        str(tmp_path),
        _model_config(),
        torch.device("cpu"),
        resolved_config,
        cpu_offload=True,
        offload_flag="image_encoder_cpu_offload",
    )

    assert _PassthroughEncoder.loaded_device == torch.device("cuda:4")
    assert resolved_config.override_log == ()
    probe.assert_called_once_with(4)
