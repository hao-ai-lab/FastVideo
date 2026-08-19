# SPDX-License-Identifier: Apache-2.0
"""Wan activation-checkpoint and regional-compile setup tests."""

from __future__ import annotations

import torch

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.train.models.wan import wan
from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.training_config import (
    ModelTrainingConfig,
    TrainingConfig,
)


def test_wan_passes_activation_checkpointing_to_pre_fsdp_load(monkeypatch) -> None:
    training_config = TrainingConfig(
        model=ModelTrainingConfig(enable_gradient_checkpointing_type="full"),
        pipeline_config=PipelineConfig(),
    )
    captured: dict = {}
    transformer = torch.nn.Linear(1, 1)

    def _fake_load_module_from_path(**kwargs):
        captured.update(kwargs)
        return transformer

    monkeypatch.setattr(wan, "load_module_from_path", _fake_load_module_from_path)
    monkeypatch.setattr(wan, "apply_trainable", lambda module, trainable: module)
    monkeypatch.setattr(WanModel, "_enable_lora_if_configured", lambda self, module: False)

    model = WanModel.__new__(WanModel)
    result = model._load_transformer(
        init_from="fake/model",
        trainable=True,
        disable_custom_init_weights=False,
        enable_gradient_checkpointing_type=None,
        training_config=training_config,
    )

    assert result is transformer
    transform = captured["pre_fsdp_transform"]
    assert transform is not None
    assert transform.keywords == {"checkpointing_type": "full"}


def test_wan_skips_activation_checkpointing_for_frozen_role(monkeypatch) -> None:
    training_config = TrainingConfig(
        model=ModelTrainingConfig(enable_gradient_checkpointing_type="full"),
        pipeline_config=PipelineConfig(),
    )
    captured: dict = {}
    transformer = torch.nn.Linear(1, 1)

    def _fake_load_module_from_path(**kwargs):
        captured.update(kwargs)
        return transformer

    monkeypatch.setattr(wan, "load_module_from_path", _fake_load_module_from_path)
    monkeypatch.setattr(wan, "apply_trainable", lambda module, trainable: module)
    monkeypatch.setattr(WanModel, "_enable_lora_if_configured", lambda self, module: False)

    model = WanModel.__new__(WanModel)
    model._load_transformer(
        init_from="fake/model",
        trainable=False,
        disable_custom_init_weights=True,
        enable_gradient_checkpointing_type=None,
        training_config=training_config,
    )

    assert captured["pre_fsdp_transform"] is None
