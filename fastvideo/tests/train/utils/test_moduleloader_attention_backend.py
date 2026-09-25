# SPDX-License-Identifier: Apache-2.0
"""Verify construction-scoped attention backends in the training loader."""

from __future__ import annotations

import torch
import pytest

from fastvideo.attention.selector import _active_component_attention_backend_scope
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.utils import moduleloader
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.training_config import (
    DistributedConfig,
    TrainingConfig,
)


def test_load_transformer_scopes_attention_backend(monkeypatch, tmp_path) -> None:
    """Apply one backend while accepting trailing modular-manifest metadata."""
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    captured: list[tuple[AttentionBackendEnum | None, str | None]] = []

    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer", {
            "subfolder": "transformer"
        })},
    )

    def _fake_load_module(**kwargs):
        del kwargs
        scope = _active_component_attention_backend_scope()
        captured.append((scope.backend, scope.component) if scope else (None, None))
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
    assert captured == [(AttentionBackendEnum.ATTN_QAT_TRAIN, "transformer")]
    assert _active_component_attention_backend_scope() is None


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
        scope = _active_component_attention_backend_scope()
        assert scope is not None and scope.backend is AttentionBackendEnum.ATTN_QAT_TRAIN
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
    assert _active_component_attention_backend_scope() is None


def test_fullattn_vsa_recipe_scopes_student_and_dense_teacher(monkeypatch, tmp_path) -> None:
    from fastvideo.models.wan.pipeline_config import FastWan2_2_TI2V_5B_Config

    recipe = "examples/train/configs/fine_tuning/wan/fast_ti2v_fullattn_lora_vsa.yaml"
    training_config = load_run_config(recipe).training
    captured = []
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )

    def _fake_load_module(**kwargs):
        scope = _active_component_attention_backend_scope()
        args = kwargs["fastvideo_args"]
        captured.append((scope.backend, type(args.pipeline_config),
                         hasattr(args, "_loading_teacher_critic_model")))
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(moduleloader.PipelineComponentLoader, "load_module", _fake_load_module)
    model_path = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
    moduleloader.load_module_from_path(model_path=model_path, module_type="transformer",
                                       training_config=training_config, attention_backend="VIDEO_SPARSE_ATTN")
    moduleloader.load_module_from_path(model_path=model_path, module_type="transformer",
                                       training_config=training_config, attention_backend="FLASH_ATTN",
                                       disable_custom_init_weights=True)
    assert captured == [
        (AttentionBackendEnum.VIDEO_SPARSE_ATTN, FastWan2_2_TI2V_5B_Config, False),
        (AttentionBackendEnum.FLASH_ATTN, FastWan2_2_TI2V_5B_Config, True),
    ]
    assert _active_component_attention_backend_scope() is None
