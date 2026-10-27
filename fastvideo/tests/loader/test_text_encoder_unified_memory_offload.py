# SPDX-License-Identifier: Apache-2.0
"""Regression tests for encoder placement on unified memory."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import ImageEncoderLoader, TextEncoderLoader


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
    args = FastVideoArgs(model_path=str(tmp_path), text_encoder_cpu_offload=True)

    model = TextEncoderLoader().load_model(
        str(tmp_path),
        _model_config(),
        requested_target,
        args,
        cpu_offload=cpu_offload,
    )

    assert isinstance(model, _PassthroughEncoder)
    assert _PassthroughEncoder.loaded_device == torch.device("cuda:5")
    assert args.text_encoder_cpu_offload is False
    probe.assert_called_once_with(5)


def test_discrete_memory_preserves_explicit_cpu_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("fastvideo.models.loader.component_loader.get_local_torch_device",
                        lambda: torch.device("cuda:2"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr(
        "fastvideo.models.loader.component_loader.ModelRegistry.resolve_model_cls",
        lambda architectures: (_PassthroughEncoder, None),
    )
    args = FastVideoArgs(model_path=str(tmp_path), text_encoder_cpu_offload=False)

    TextEncoderLoader().load_model(
        str(tmp_path),
        _model_config(),
        torch.device("cuda:2"),
        args,
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
    args = FastVideoArgs(
        model_path=str(tmp_path),
        text_encoder_cpu_offload=True,
        image_encoder_cpu_offload=True,
    )

    ImageEncoderLoader().load_model(
        str(tmp_path),
        _model_config(),
        torch.device("cpu"),
        args,
        cpu_offload=True,
        offload_flag="image_encoder_cpu_offload",
    )

    assert _PassthroughEncoder.loaded_device == torch.device("cuda:4")
    assert args.text_encoder_cpu_offload is False
    assert args.image_encoder_cpu_offload is False
    probe.assert_called_once_with(4)
