# SPDX-License-Identifier: Apache-2.0
"""FastH3 Parallel Decoding Distillation (PDD) exports' ``fastvideo_inference.json``.

A PDD contract names the fused-block partition of the transformer's widened
heads and, for Ref2VA, the reference-video VSA policy. The pipeline applies
both to its config and rejects any contract it cannot honor exactly. DMD
contracts are covered by test_minimax_h3_distilled_schedule.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fastvideo import envs
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
    MiniMaxH3ModularPipeline,
    MiniMaxH3Ref2VAModularPipeline,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

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
    # The export was trained with VSA-H3; tests that probe the backend pass their own.
    args.setdefault("attention_backend", "VIDEO_SPARSE_ATTN_H3")
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


@pytest.mark.parametrize("indices", [list(GRID32_BLOCKS8), tuple(GRID32_BLOCKS8)])
def test_matching_explicit_settings_are_accepted(tmp_path, indices):
    config = MiniMaxH3PipelineConfig(pdd_step_indices=indices,
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


def test_a_dmd_contract_sets_its_ladder_on_the_ref2va_pipeline(tmp_path):
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


@pytest.mark.parametrize("field,value,match", [
    ("unexpected_knob", 1, "unsupported keys"),
    ("dmd_denoising_steps", [999, 500], "unsupported keys"),
    ("schema", "fasth3-inference-contract-v2", "disagrees with its schema_version"),
    ("model_type", "t2va", "is a 't2va' PDD export"),
    ("transformer_component", "transformer", "targets 'transformer'"),
    ("conditioning", "noised_references", "Unsupported FastH3 Ref2VA conditioning"),
    ("base_model_revision", "", "base_model_revision must be a non-empty string"),
    ("pdd_steps", 1, "pdd_steps must be an int >= 2"),
    ("pdd_steps", True, "pdd_steps must be an int >= 2"),
    ("pdd_step_indices", _MISSING, "pdd_step_indices must be a list of integers"),
    ("pdd_step_indices", [0, 4, 8, 12, 16, 20, 24, 28, 31], "must start at 0 and end at 32"),
    ("pdd_step_indices", [1, 4, 8, 12, 16, 20, 24, 28, 32], "must start at 0 and end at 32"),
    ("pdd_step_indices", [0, 8, 4, 12, 16, 20, 24, 28, 32], "must be strictly increasing"),
    ("pdd_step_indices", [0.0, 4, 8, 12, 16, 20, 24, 28, 32], "pdd_step_indices must be a list of integers"),
    ("pdd_step_indices", [False, 4, 8, 12, 16, 20, 24, 28, 32], "pdd_step_indices must be a list of integers"),
    ("pdd_step_indices", [32], "must name at least one block"),
    ("num_inference_steps", 9, "num_inference_steps=9 must equal the 8 fused blocks"),
    ("num_inference_steps", _MISSING, "num_inference_steps=None must equal"),
    ("transformer_forwards", 4, "transformer_forwards=4 must equal"),
    ("transformer_forwards", 8.0, "transformer_forwards=8.0 must equal"),
    ("grid_max_t", 1.0, "grid_max_t=1.0 is unsupported"),
    ("video_scheduler_shift", _MISSING, "video_scheduler_shift=None disagrees"),
    ("video_scheduler_shift", float("nan"), "video_scheduler_shift=nan disagrees"),
    ("audio_scheduler_shift", True, "audio_scheduler_shift=True disagrees"),
    ("audio_scheduler_shift", 4.0, "audio_scheduler_shift=4.0 disagrees"),
    ("guidance_scale", 2.0, "guidance_scale=2.0 is unsupported"),
    ("attention_backend", "NOT_A_BACKEND", "Unknown attention backend"),
    ("vsa_sparsity", 1.0, r"vsa_sparsity must be in \[0, 1\)"),
    ("vsa_tile_size", 96, "vsa_tile_size must be one of"),
    ("vsa_tile_size", 128.0, "vsa_tile_size must be one of"),
    ("vsa_ref_policy", "p1", "Unsupported FastH3 PDD vsa_ref_policy 'p1'"),
    ("vsa_ref_policy", _MISSING, "vsa_ref_keep_rate requires vsa_ref_policy"),
    ("vsa_ref_keep_rate", 0.0, r"vsa_ref_keep_rate must be in \(0, 1\), got 0.0"),
    ("vsa_ref_keep_rate", 1.0, r"vsa_ref_keep_rate must be in \(0, 1\), got 1.0"),
    ("vsa_ref_keep_rate", True, r"vsa_ref_keep_rate must be in \(0, 1\), got True"),
    ("vsa_ref_keep_rate", _MISSING, r"vsa_ref_keep_rate must be in \(0, 1\), got None"),
])
def test_malformed_pdd_contracts_are_rejected(tmp_path, field, value, match):
    contract = dict(PDD_CONTRACT)
    if value is _MISSING:
        contract.pop(field)
    else:
        contract[field] = value
    with pytest.raises((ValueError, TypeError), match=match):
        _initialize(_pipeline(tmp_path, contract))


def test_trained_sparsity_or_tile_size_mismatch_is_logged(tmp_path, monkeypatch):
    from fastvideo.pipelines.basic.minimax_h3 import minimax_h3_pipeline

    warnings = []
    monkeypatch.setattr(minimax_h3_pipeline.logger, "warning", lambda message, *args: warnings.append(message % args))
    _initialize(_pipeline(tmp_path, PDD_CONTRACT),
                attention_backend="VIDEO_SPARSE_ATTN_H3",
                VSA_sparsity=0.9,
                VSA_tile_size=256)
    assert warnings == ["FastH3 PDD checkpoint was trained with VSA_tile_size=128; this run uses 256."]


_BACKEND_ERROR = ("needs attention_backend=VIDEO_SPARSE_ATTN_H3: its fastvideo_inference.json was trained with "
                  "VIDEO_SPARSE_ATTN_H3.*This run requests {requested}. Select VIDEO_SPARSE_ATTN_H3 with "
                  "VSA_sparsity=0.9, VSA_tile_size=128")
_GATE = "transformer_blocks.0.attn.to_gate_compress.weight"


def _write_gates(transformer_dir, indexed=True):
    if indexed:
        (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {_GATE: "diffusion_pytorch_model-00001-of-00001.safetensors"}}))
    else:
        save_file({_GATE: torch.zeros(2, 2)}, str(transformer_dir / "diffusion_pytorch_model.safetensors"))


@pytest.mark.parametrize("requested,shown", [("FLASH_ATTN", "FLASH_ATTN"), (None, "automatic selection")])
def test_a_vsa_trained_export_rejects_other_attention_backends(tmp_path, env_overrides, requested, shown):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))
    with pytest.raises(ValueError, match=_BACKEND_ERROR.format(requested=shown)):
        _initialize(_pipeline(tmp_path, PDD_CONTRACT), attention_backend=requested)


def test_the_environment_backend_counts_as_the_request(tmp_path, env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("VIDEO_SPARSE_ATTN_H3"))
    config = _initialize(_pipeline(tmp_path, PDD_CONTRACT), attention_backend=None)
    assert config.vsa_ref_policy == "p2_multi_region"


_MINIMAL_KEYS = ("schema_version", "pdd_steps", "pdd_step_indices", "num_inference_steps", "transformer_forwards",
                 "video_scheduler_shift", "audio_scheduler_shift")


@pytest.mark.parametrize("indexed", [True, False])
def test_trained_gates_require_vsa_even_without_a_contract_backend(tmp_path, env_overrides, indexed):
    """The checkpoint's own to_gate_compress weights mark it as a VSA-H3 student."""
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))
    minimal = {key: PDD_CONTRACT[key] for key in _MINIMAL_KEYS}
    pipeline = _pipeline(tmp_path, minimal)
    _write_gates(tmp_path / "transformer_ref", indexed)
    with pytest.raises(ValueError, match="its transformer_ref carries VSA compression gates"):
        _initialize(pipeline, attention_backend="FLASH_ATTN")
    _initialize(pipeline, attention_backend="VIDEO_SPARSE_ATTN_H3")


def test_a_minimal_contract_without_gates_leaves_the_backend_to_the_run(tmp_path, env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))
    minimal = {key: PDD_CONTRACT[key] for key in _MINIMAL_KEYS}
    config = _initialize(_pipeline(tmp_path, minimal), attention_backend="FLASH_ATTN")
    assert config.pdd_step_indices == GRID32_BLOCKS8


@pytest.mark.parametrize("recorded,requested,accepted", [
    ("VIDEO_SPARSE_ATTN_H3", None, True),
    ("FLASH_ATTN", "VIDEO_SPARSE_ATTN_H3", False),
])
def test_a_loaded_transformer_reports_its_own_backend(tmp_path, env_overrides, recorded, requested, accepted):
    """A transformer built elsewhere (for example by a trainer) carries the backend it was built with."""
    from fastvideo.platforms import AttentionBackendEnum

    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))
    pipeline = _pipeline(tmp_path, PDD_CONTRACT)
    pipeline.modules["transformer"] = SimpleNamespace(config=SimpleNamespace(
        _resolved_attention_backend=AttentionBackendEnum[recorded]))
    if accepted:
        _initialize(pipeline, attention_backend=requested)
    else:
        with pytest.raises(ValueError, match="This run requests FLASH_ATTN"):
            _initialize(pipeline, attention_backend=requested)


@pytest.fixture
def hub_snapshot(tmp_path, monkeypatch):
    """Resolve the Hub repo id "org/fasth3" to tmp_path, as maybe_download_model does after downloading."""
    from fastvideo.pipelines import composed_pipeline_base

    downloads = []

    def download(model_path, **kwargs):
        downloads.append(model_path)
        return str(tmp_path) if model_path == "org/fasth3" else model_path

    monkeypatch.setattr(composed_pipeline_base, "maybe_download_model", download)
    monkeypatch.setattr(composed_pipeline_base, "verify_model_config_and_directory",
                        lambda model_path, **kwargs: {"_class_name": "MiniMaxH3ModularPipeline"})
    return downloads


@pytest.mark.parametrize("contract", [PDD_CONTRACT, DMD_CONTRACT, None], ids=["pdd", "dmd", "no-contract"])
def test_config_load_rejects_the_backend_for_any_gated_checkpoint(tmp_path, env_overrides, hub_snapshot, contract):
    """Checked once the Hub snapshot resolves, before any component loads, whatever the schedule."""
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))
    pipeline = _pipeline(tmp_path, contract or {}, pdd_steps=32 if contract is PDD_CONTRACT else None)
    if contract is None:
        (tmp_path / "fastvideo_inference.json").unlink()
    _write_gates(tmp_path / "transformer_ref")
    pipeline.model_path = "org/fasth3"
    pipeline.fastvideo_args = SimpleNamespace(attention_backend="FLASH_ATTN")
    with pytest.raises(ValueError, match="its transformer_ref carries VSA compression gates.*requests FLASH_ATTN"):
        pipeline._load_config(pipeline.model_path)
    assert hub_snapshot == ["org/fasth3"] and pipeline.model_path == str(tmp_path)
    pipeline.fastvideo_args = SimpleNamespace(attention_backend="VIDEO_SPARSE_ATTN_H3")
    assert pipeline._load_config(pipeline.model_path) == {"_class_name": "MiniMaxH3ModularPipeline"}


def test_load_modules_checks_before_loading_unless_the_transformer_is_supplied(tmp_path, env_overrides, monkeypatch,
                                                                               hub_snapshot):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))
    pipeline = _pipeline(tmp_path, PDD_CONTRACT)
    pipeline._ref2va = True
    loads = []

    def base_load_modules(self, fastvideo_args, loaded_modules=None):
        self._load_config(self.model_path)
        loads.append(loaded_modules)
        return {}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", base_load_modules)
    args = SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig(), attention_backend="FLASH_ATTN",
                           inference_mode=False)
    pipeline.fastvideo_args = args
    with pytest.raises(ValueError, match="needs attention_backend=VIDEO_SPARSE_ATTN_H3"):
        pipeline.load_modules(args)
    assert loads == []
    # A transformer the caller already built is not rebuilt, so its checkpoint is not checked.
    supplied = {"transformer": object()}
    pipeline.load_modules(args, loaded_modules=supplied)
    assert loads == [supplied]


def test_the_contract_is_read_once_per_resolved_path(tmp_path, monkeypatch, hub_snapshot):
    from fastvideo.pipelines.basic.minimax_h3 import minimax_h3_pipeline

    reads = []
    monkeypatch.setattr(minimax_h3_pipeline, "json",
                        SimpleNamespace(loads=lambda text: reads.append(text) or json.loads(text), dumps=json.dumps))
    pipeline = _pipeline(tmp_path, PDD_CONTRACT)
    pipeline.model_path = "org/fasth3"
    pipeline.fastvideo_args = SimpleNamespace(attention_backend="VIDEO_SPARSE_ATTN_H3")
    pipeline._load_config(pipeline.model_path)
    _initialize(pipeline)
    assert len(reads) == 1




def test_reference_policy_requires_vsa(tmp_path, env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))
    config = MiniMaxH3PipelineConfig(vsa_ref_policy="p2_multi_region", vsa_ref_keep_rate=0.1)
    with pytest.raises(ValueError, match="vsa_ref_policy='p2_multi_region' sparsifies reference videos"):
        _initialize(_pipeline(tmp_path, DMD_CONTRACT, pdd_steps=None), config, attention_backend="FLASH_ATTN")


def test_forward_checks_the_fused_block_count_before_conditioning(tmp_path):
    pipeline = _pipeline(tmp_path, PDD_CONTRACT)
    config = _initialize(pipeline)
    pipeline.post_init_called = True
    batch = ForwardBatch(data_type="video", num_inference_steps=4)
    with pytest.raises(ValueError, match="num_inference_steps counts transformer forwards and must be 8, got 4"):
        pipeline.forward(batch, SimpleNamespace(pipeline_config=config))
