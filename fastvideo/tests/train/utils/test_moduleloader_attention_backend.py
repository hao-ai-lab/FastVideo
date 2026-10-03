# SPDX-License-Identifier: Apache-2.0
"""Verify construction-scoped attention backends and resolved configs in the training loader."""

from __future__ import annotations

import torch
import pytest

from fastvideo.api.schema import ExecutionMode
from fastvideo.attention.selector import _active_component_attention_backend_scope
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.utils import moduleloader
from fastvideo.train.utils.training_config import (
    DistributedConfig,
    TrainingConfig,
)

# A registered model path, so resolution finds its PipelineConfig class without a download.
_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


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
        model_path=_MODEL_PATH,
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
            model_path=_MODEL_PATH,
            module_type="transformer",
            training_config=training_config,
            attention_backend="ATTN_QAT_TRAIN",
        )
    assert _active_component_attention_backend_scope() is None


def _wan_training_config() -> TrainingConfig:
    """Training config with the Wan model definition and a two-GPU sequence-parallel layout."""
    return TrainingConfig(
        distributed=DistributedConfig(num_gpus=2, sp_size=2, hsdp_shard_dim=2),
        pipeline_config=PipelineConfig.from_kwargs({"model_path": _MODEL_PATH}),
        model_path=_MODEL_PATH,
        vsa_sparsity=0.5,
    )


def test_load_module_passes_distillation_config_and_teacher_flag(monkeypatch, tmp_path) -> None:
    """The loader receives the transformer overrides as typed values and the teacher/critic flag as a keyword."""
    training_config = _wan_training_config()
    captured: dict = {}
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "WanTransformer3DModel")},
    )

    def _fake_load_module(**kwargs):
        captured.update(kwargs)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(moduleloader.PipelineComponentLoader, "load_module", _fake_load_module)

    moduleloader.load_module_from_path(
        model_path=_MODEL_PATH,
        module_type="transformer",
        training_config=training_config,
        disable_custom_init_weights=True,
        override_transformer_cls_name="CausalWanTransformer3DModel",
        transformer_override_safetensor="/weights/student.safetensors",
    )

    resolved_config = captured["resolved_config"]
    assert captured["loading_teacher_critic_model"] is True
    assert resolved_config.mode is ExecutionMode.DISTILLATION
    assert resolved_config.training_mode
    assert resolved_config.model_path == _MODEL_PATH
    assert resolved_config.pipeline.components.override_transformer_cls_name == "CausalWanTransformer3DModel"
    assert resolved_config.pipeline.components.transformer_weights == "/weights/student.safetensors"
    assert resolved_config.engine.num_gpus == 2
    assert resolved_config.engine.parallelism.sp_size == 2
    assert resolved_config.engine.parallelism.hsdp_shard_dim == 2
    assert resolved_config.engine.precision.dit == "fp32"
    assert not any((resolved_config.engine.offload.dit, resolved_config.engine.offload.dit_layerwise,
                    resolved_config.engine.offload.text_encoder, resolved_config.engine.offload.vae))
    assert type(resolved_config.pipeline_config) is type(training_config.pipeline_config)
    assert resolved_config.pipeline_config is not training_config.pipeline_config


def test_vae_load_keeps_checkpoint_vae_config(monkeypatch, tmp_path) -> None:
    """The VAE config that the loader filled from the checkpoint becomes the training pipeline config's VAE config."""
    training_config = _wan_training_config()
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"vae": ("diffusers", "AutoencoderKLWan")},
    )

    def _fill_vae_config(**kwargs):
        kwargs["resolved_config"].pipeline_config.vae_config.arch_config.z_dim = 48
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(moduleloader.PipelineComponentLoader, "load_module", _fill_vae_config)

    moduleloader.load_module_from_path(
        model_path=_MODEL_PATH,
        module_type="vae",
        training_config=training_config,
    )

    assert training_config.pipeline_config.vae_config.arch_config.z_dim == 48


def test_inference_resolved_config_offloads_only_the_dit() -> None:
    """The inference config keeps encoders and VAE on the device and carries the training VSA sparsity."""
    resolved_config = moduleloader.build_inference_resolved_config(_wan_training_config(), model_path=_MODEL_PATH)

    assert resolved_config.mode is ExecutionMode.INFERENCE
    assert resolved_config.inference_mode
    assert resolved_config.engine.offload.dit is True
    assert resolved_config.engine.offload.text_encoder is False
    assert resolved_config.engine.offload.vae is False
    assert resolved_config.engine.attention.vsa_sparsity == 0.5
    assert resolved_config.engine.precision.dit == "fp32"
