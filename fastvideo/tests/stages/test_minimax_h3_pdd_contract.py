# SPDX-License-Identifier: Apache-2.0
"""FastH3 Parallel Decoding Distillation (PDD) exports' ``fastvideo_inference.json``.

A PDD contract names the fused-block partition of the transformer's widened
heads and, for Ref2VA, the reference-video VSA policy. The pipeline applies
both to its config and rejects any contract it cannot honor exactly; the DMD
contract keeps its existing behavior (see test_minimax_h3_distilled_schedule).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
    MiniMaxH3ModularPipeline,
    MiniMaxH3Ref2VAModularPipeline,
)

GRID32_BLOCKS8 = list(range(0, 33, 4))
# The OmniRef PDD-8 export's contract, verbatim.
PDD_CONTRACT = {
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "audio_scheduler_shift": 3.0,
    "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da",
    "conditioning": "fixed_ordered_references_target_only_flow",
    "grid_max_t": 0.999,
    "guidance_scale": 1.0,
    "model_type": "ref2va",
    "num_inference_steps": 8,
    "pdd_step_indices": GRID32_BLOCKS8,
    "pdd_steps": 32,
    "schema": "fasth3-inference-contract-v1",
    "schema_version": "fasth3-inference-contract-v1",
    "transformer_component": "transformer_ref",
    "transformer_forwards": 8,
    "video_scheduler_shift": 12.0,
    "vsa_ref_keep_rate": 0.1,
    "vsa_ref_policy": "p2_multi_region",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
}
DMD_CONTRACT = {
    "schema_version": "fasth3-inference-contract-v1",
    "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
    "num_inference_steps": 9,
    "transformer_forwards": 8,
    "video_scheduler_shift": 12.0,
    "audio_scheduler_shift": 3.0,
}


def _pipeline(tmp_path, contract, *, cls=MiniMaxH3Ref2VAModularPipeline, pdd_steps=32, video_shift=12.0):
    """Initialization with weight loading omitted: schedulers, transformer config.json, and the sidecar."""
    transformer_dir = tmp_path / cls._extra_config_module_map.get("transformer", "transformer")
    transformer_dir.mkdir(exist_ok=True)
    transformer_config = {"_class_name": "MiniMaxH3Transformer3DModel", "_diffusers_version": "0.36.0.dev0"}
    if pdd_steps is not None:
        transformer_config["pdd_steps"] = pdd_steps
    (transformer_dir / "config.json").write_text(json.dumps(transformer_config))
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(contract))
    pipeline = object.__new__(cls)
    pipeline.model_path = str(tmp_path)
    pipeline.modules = {
        "scheduler": MiniMaxH3Scheduler(shift=video_shift),
        "audio_scheduler": MiniMaxH3Scheduler(shift=3.0),
    }
    return pipeline


def _initialize(pipeline, config=None, **args):
    config = config if config is not None else MiniMaxH3PipelineConfig()
    pipeline.initialize_pipeline(SimpleNamespace(pipeline_config=config, **args))
    return config


def test_pdd_contract_sets_the_trained_partition_and_reference_policy(tmp_path):
    config = _initialize(_pipeline(tmp_path, PDD_CONTRACT))
    assert config.dit_config.arch_config.pdd_steps == 32
    assert config.pdd_step_indices == GRID32_BLOCKS8
    assert config.vsa_ref_policy == "p2_multi_region"
    assert config.vsa_ref_keep_rate == 0.1
    assert config.dmd_denoising_steps is None
    config.check_pipeline_config()


def test_minimal_pdd_contract_without_optional_fields(tmp_path):
    required = {
        key: PDD_CONTRACT[key]
        for key in ("schema_version", "pdd_steps", "pdd_step_indices", "num_inference_steps",
                    "transformer_forwards", "video_scheduler_shift", "audio_scheduler_shift")
    }
    config = _initialize(_pipeline(tmp_path, required))
    assert config.pdd_step_indices == GRID32_BLOCKS8
    assert config.vsa_ref_policy is None and config.vsa_ref_keep_rate is None


def test_matching_explicit_settings_are_accepted(tmp_path):
    config = MiniMaxH3PipelineConfig(pdd_step_indices=list(GRID32_BLOCKS8),
                                     vsa_ref_policy="p2_multi_region",
                                     vsa_ref_keep_rate=0.1)
    _initialize(_pipeline(tmp_path, PDD_CONTRACT), config)
    assert config.pdd_step_indices == GRID32_BLOCKS8


@pytest.mark.parametrize("field,value", [
    ("pdd_step_indices", [0, 8, 16, 24, 32]),
    ("vsa_ref_keep_rate", 0.2),
    ("dmd_denoising_steps", [999, 749, 500, 250]),
])
def test_conflicting_explicit_settings_are_rejected(tmp_path, field, value):
    config = MiniMaxH3PipelineConfig(**{field: value}, **({"vsa_ref_policy": "p2_multi_region"}
                                                          if field == "vsa_ref_keep_rate" else {}))
    with pytest.raises(ValueError, match="disagrees|must be unset"):
        _initialize(_pipeline(tmp_path, PDD_CONTRACT), config)


def test_dmd_contract_is_unchanged_on_the_ref2va_pipeline(tmp_path):
    config = _initialize(_pipeline(tmp_path, DMD_CONTRACT, pdd_steps=None))
    assert config.dmd_denoising_steps == DMD_CONTRACT["dmd_denoising_steps"]
    assert config.pdd_step_indices is None and config.vsa_ref_policy is None


def test_t2va_pipeline_rejects_a_ref2va_export(tmp_path):
    with pytest.raises(ValueError, match="MiniMaxH3Ref2VAModularPipeline"):
        _initialize(_pipeline(tmp_path, {**PDD_CONTRACT, "transformer_component": "transformer"},
                              cls=MiniMaxH3ModularPipeline))


def test_transformer_without_widened_heads_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="config.json pdd_steps=None"):
        _initialize(_pipeline(tmp_path, PDD_CONTRACT, pdd_steps=None))


def test_transformer_with_a_different_grid_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="disagrees with transformer_ref/config.json"):
        _initialize(_pipeline(tmp_path, PDD_CONTRACT, pdd_steps=16))


def test_video_shift_must_match_the_contract(tmp_path):
    # (tmp_path must not contain "scheduler": get_diffusers_config keys its file name on that substring)
    with pytest.raises(ValueError, match="video_scheduler_shift"):
        _initialize(_pipeline(tmp_path, PDD_CONTRACT, video_shift=10.0))


_MISSING = object()


@pytest.mark.parametrize("field,value", [
    ("unexpected_knob", 1),
    ("dmd_denoising_steps", [999, 500]),
    ("schema", "fasth3-inference-contract-v2"),
    ("model_type", "t2va"),
    ("transformer_component", "transformer"),
    ("conditioning", "noised_references"),
    ("base_model_revision", ""),
    ("pdd_steps", 1),
    ("pdd_steps", True),
    ("pdd_step_indices", _MISSING),
    ("pdd_step_indices", [0, 4, 8, 12, 16, 20, 24, 28, 31]),
    ("pdd_step_indices", [1, 4, 8, 12, 16, 20, 24, 28, 32]),
    ("pdd_step_indices", [0, 8, 4, 12, 16, 20, 24, 28, 32]),
    ("pdd_step_indices", [0.0, 4, 8, 12, 16, 20, 24, 28, 32]),
    ("pdd_step_indices", [False, 4, 8, 12, 16, 20, 24, 28, 32]),
    ("pdd_step_indices", [32]),
    ("num_inference_steps", 9),
    ("num_inference_steps", _MISSING),
    ("transformer_forwards", 4),
    ("transformer_forwards", 8.0),
    ("grid_max_t", 1.0),
    ("video_scheduler_shift", _MISSING),
    ("video_scheduler_shift", float("nan")),
    ("audio_scheduler_shift", True),
    ("audio_scheduler_shift", 4.0),
    ("guidance_scale", 2.0),
    ("attention_backend", "NOT_A_BACKEND"),
    ("vsa_sparsity", 1.0),
    ("vsa_tile_size", 96),
    ("vsa_tile_size", 128.0),
    ("vsa_ref_policy", "p1"),
    ("vsa_ref_policy", _MISSING),
    ("vsa_ref_keep_rate", 0.0),
    ("vsa_ref_keep_rate", 1.0),
    ("vsa_ref_keep_rate", True),
    ("vsa_ref_keep_rate", _MISSING),
])
def test_malformed_pdd_contracts_are_rejected(tmp_path, field, value):
    contract = dict(PDD_CONTRACT)
    if value is _MISSING:
        contract.pop(field)
    else:
        contract[field] = value
    with pytest.raises((ValueError, TypeError)):
        _initialize(_pipeline(tmp_path, contract))


def test_trained_attention_mismatch_is_logged(tmp_path, monkeypatch):
    from fastvideo.pipelines.basic.minimax_h3 import minimax_h3_pipeline

    warnings = []
    monkeypatch.setattr(minimax_h3_pipeline.logger, "warning", lambda message, *args: warnings.append(message % args))
    _initialize(_pipeline(tmp_path, PDD_CONTRACT),
                attention_backend="VIDEO_SPARSE_ATTN_H3",
                VSA_sparsity=0.9,
                VSA_tile_size=256)
    assert warnings == ["FastH3 PDD checkpoint was trained with VSA_tile_size=128; this run uses 256."]

    warnings.clear()
    _initialize(_pipeline(tmp_path, PDD_CONTRACT), attention_backend="FLASH_ATTN", VSA_sparsity=0.0,
                VSA_tile_size=256)
    assert warnings == ["FastH3 PDD checkpoint was trained with VIDEO_SPARSE_ATTN_H3 attention; this run requests "
                        "FLASH_ATTN."]
