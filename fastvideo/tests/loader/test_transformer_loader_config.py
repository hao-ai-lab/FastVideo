# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the resolved-config values that TransformerLoader hands to the FSDP model loader."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

import fastvideo.models.loader.component_loader as component_loader
from fastvideo.api.inference_resolution import resolve_inference_config, torch_compile_kwargs
from fastvideo.models.loader.component_loader import PipelineComponentLoader, TransformerLoader
from fastvideo.tests.api.config_snapshot import isolated_environment

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class _FakeTransformer(nn.Module):
    """Model class that the registry returns for the test checkpoint."""


def _resolve(raw: dict):
    """A resolved inference config for a registered Wan model with ``raw`` merged into its input."""
    with isolated_environment():
        return resolve_inference_config({"model_path": WAN_T2V, **raw})


@pytest.fixture
def transformer_dir(tmp_path):
    """A transformer component directory with a Diffusers config and one weight file."""
    component = tmp_path / "transformer"
    component.mkdir()
    (component / "config.json").write_text(json.dumps({"_class_name": "WanTransformer3DModel"}))
    (component / "model.safetensors").write_bytes(b"")
    return component


@pytest.fixture
def fsdp_calls(monkeypatch):
    """Keyword arguments of each ``maybe_load_fsdp_model`` call; each call returns a one-layer model."""
    calls = []

    def fake_maybe_load_fsdp_model(**kwargs):
        calls.append(kwargs)
        return nn.Linear(1, 1).to(kwargs["default_dtype"])

    monkeypatch.setattr(component_loader, "maybe_load_fsdp_model", fake_maybe_load_fsdp_model)
    monkeypatch.setattr(component_loader.ModelRegistry, "resolve_model_cls", lambda name: (_FakeTransformer, None))
    monkeypatch.setattr(component_loader, "get_local_torch_device", lambda: torch.device("cpu"))
    return calls


def test_transformer_loader_passes_the_typed_engine_settings(transformer_dir, fsdp_calls) -> None:
    resolved_config = _resolve({
        "engine": {
            "use_fsdp_inference": True,
            "offload": {
                "dit": False,
                "pin_cpu_memory": False
            },
            "compile": {
                "enabled": True,
                "mode": "max-autotune",
                "extras": {
                    "dynamic": False
                }
            },
            "attention": {
                "vsa_tile_size": 128
            },
            "parallelism": {
                "hsdp_shard_dim": 1
            },
        },
        "pipeline": {
            "components": {
                "lora_path": "/adapters/style",
                "lora_strength": 0.5
            }
        },
    })

    TransformerLoader().load(str(transformer_dir), resolved_config)

    (call, ) = fsdp_calls
    assert call["weight_dir_list"] == [str(transformer_dir / "model.safetensors")]
    assert call["default_dtype"] is torch.bfloat16
    assert (call["hsdp_replicate_dim"], call["hsdp_shard_dim"]) == (1, 1)
    assert (call["cpu_offload"], call["pin_cpu_memory"], call["fsdp_inference"]) == (False, False, True)
    assert call["training_mode"] is False
    assert call["enable_torch_compile"] is True
    assert call["torch_compile_kwargs"] == torch_compile_kwargs(resolved_config) == {
        "mode": "max-autotune",
        "dynamic": False
    }
    assert call["inference_regional_compile"] is False
    assert call["inference_vsa_tile_size"] == 128
    assert (call["lora_path"], call["lora_strength"]) == ("/adapters/style", 0.5)


def test_teacher_critic_flag_loads_checkpoint_weights_without_quantization(monkeypatch, tmp_path, transformer_dir,
                                                                          fsdp_calls) -> None:
    generator_weights = tmp_path / "generator.safetensors"
    generator_weights.write_bytes(b"")
    resolved_config = _resolve({"pipeline": {"components": {"transformer_weights": str(generator_weights)}}})
    quant_config = object()
    monkeypatch.setattr(resolved_config.pipeline_config.dit_config, "quant_config", quant_config)

    PipelineComponentLoader.load_module("transformer", str(transformer_dir), "diffusers", resolved_config)
    PipelineComponentLoader.load_module("transformer",
                                        str(transformer_dir),
                                        "diffusers",
                                        resolved_config,
                                        loading_teacher_critic_model=True)

    generator, teacher = fsdp_calls
    assert generator["weight_dir_list"] == [str(generator_weights)]
    assert generator["init_params"]["config"].quant_config is not None
    assert teacher["weight_dir_list"] == [str(transformer_dir / "model.safetensors")]
    assert teacher["init_params"]["config"].quant_config is None
    assert resolved_config.pipeline_config.dit_config.quant_config is quant_config
