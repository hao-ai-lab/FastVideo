# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import pytest

from fastvideo.attention.selector import get_global_forced_attn_backend
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.utils import moduleloader
from fastvideo.train.utils.training_config import (
    DistributedConfig,
    TrainingConfig,
)


def test_load_transformer_scopes_attention_backend(monkeypatch, tmp_path) -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    captured: list[AttentionBackendEnum | None] = []

    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )

    def _fake_load_module(**kwargs):
        del kwargs
        captured.append(get_global_forced_attn_backend())
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(
        moduleloader.PipelineComponentLoader,
        "load_module",
        _fake_load_module,
    )

    result = moduleloader.load_module_from_path(
        model_path="fake/model",
        module_type="transformer",
        training_config=training_config,
        attention_backend="ATTN_QAT_TRAIN",
    )

    assert isinstance(result, torch.nn.Module)
    assert captured == [AttentionBackendEnum.ATTN_QAT_TRAIN]
    assert get_global_forced_attn_backend() is None


def test_load_transformer_restores_backend_when_loading_fails(
    monkeypatch,
    tmp_path,
) -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )

    def _raise_during_load(**kwargs):
        del kwargs
        assert get_global_forced_attn_backend() is AttentionBackendEnum.ATTN_QAT_TRAIN
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        moduleloader.PipelineComponentLoader,
        "load_module",
        _raise_during_load,
    )

    with pytest.raises(RuntimeError, match="load failed"):
        moduleloader.load_module_from_path(
            model_path="fake/model",
            module_type="transformer",
            training_config=training_config,
            attention_backend="ATTN_QAT_TRAIN",
        )
    assert get_global_forced_attn_backend() is None
