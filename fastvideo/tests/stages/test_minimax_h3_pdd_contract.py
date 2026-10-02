# SPDX-License-Identifier: Apache-2.0
"""How a FastH3 Ref2VA PDD checkpoint's ``fastvideo_inference.json`` sets a run's settings.

``MiniMaxH3PipelineConfig.resolve_checkpoint_settings`` runs inside
``FastVideoArgs.__post_init__``. It fills each unset run setting from the file,
rejects an explicit value that differs for a setting fixed by training (attention
backend, VSA tile size, PDD partition), keeps an explicit value that differs for
a tunable setting (VSA sparsity, reference keep rate) with a warning, and
rejects a file that is incomplete or disagrees with the checkpoint file that owns
a value. ``MiniMaxH3PipelineConfig.apply_request_constraints`` then fills an
unset request step count with the block count and rejects a different one. In
the worker, ``MiniMaxH3BasePipeline`` checks its transformer before loading
weights: against those settings, and, when the transformer carries VSA
compression gates, against the run's attention backend. A DMD export's schedule
comes from its ``dmd_denoising_steps`` (see test_minimax_h3_distilled_schedule.py).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fastvideo import envs
from fastvideo.api.compat import normalize_generation_request
from fastvideo.api.parser import parse_config
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.api.schema import GenerationRequest
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
    MiniMaxH3ModularPipeline,
    MiniMaxH3Ref2VAModularPipeline,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

GRID32_BLOCKS8 = [0, 4, 8, 12, 16, 20, 24, 28, 32]
# The OmniRef PDD-8 export's fastvideo_inference.json, verbatim.
PDD_FILE = {
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
# A DMD export's fastvideo_inference.json, which has no PDD fields.
DMD_FILE = {
    "schema_version": "fasth3-inference-contract-v1",
    "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
    "num_inference_steps": 9,
    "transformer_forwards": 8,
    "video_scheduler_shift": 12.0,
    "audio_scheduler_shift": 3.0,
}


@pytest.fixture(autouse=True)
def _unset_backend_env_var(env_overrides):
    """Start every test without FASTVIDEO_ATTENTION_BACKEND, which FastVideoArgs reads as an explicit backend."""
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))


def _checkpoint(root: Path, inference_file: dict | None, *, pdd_steps: int | None = 32) -> str:
    """Write a Ref2VA checkpoint's JSON files: the inference file and the configs that own its repeated values.

    The transformer_ref config.json sets ``pdd_steps`` unless it is None; the
    scheduler configs hold the video/audio shifts 12.0/3.0. No weights are written.
    """
    files = {
        "transformer_ref/config.json": {} if pdd_steps is None else {"pdd_steps": pdd_steps},
        "scheduler/scheduler_config.json": {"shift": 12.0},
        "audio_scheduler/scheduler_config.json": {"shift": 3.0},
    }
    if inference_file is not None:
        files["fastvideo_inference.json"] = inference_file
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    return str(root)


def _run_args(model_path: str, config_settings: dict | None = None, **run_settings) -> FastVideoArgs:
    """Build a run's FastVideoArgs; construction resolves the checkpoint's settings."""
    pipeline_config = MiniMaxH3PipelineConfig(**(config_settings or {}))
    return FastVideoArgs(model_path=model_path, pipeline_config=pipeline_config, **run_settings)


def _pipeline(model_path: str, cls=MiniMaxH3Ref2VAModularPipeline):
    """A pipeline without loaded weights: the checkpoint path, its schedulers, and the Ref2VA flag __init__ sets."""
    pipeline = object.__new__(cls)
    pipeline.model_path = model_path
    pipeline._ref2va = cls._ref2va_default
    pipeline.modules = {
        "scheduler": MiniMaxH3Scheduler(shift=12.0),
        "audio_scheduler": MiniMaxH3Scheduler(shift=3.0),
    }
    return pipeline


# ------------------------------------------------- resolve_checkpoint_settings: run settings on a PDD checkpoint


def test_resolve_checkpoint_settings_unset_run_settings(tmp_path):
    args = _run_args(_checkpoint(tmp_path, PDD_FILE))
    assert args.attention_backend == "VIDEO_SPARSE_ATTN_H3"
    assert args.VSA_tile_size == 128
    assert args.VSA_sparsity == 0.9
    assert args.pipeline_config.pdd_step_indices == tuple(GRID32_BLOCKS8)
    assert args.pipeline_config.vsa_ref_keep_rate == 0.1


def test_resolve_checkpoint_settings_matching_explicit_settings(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        args = _run_args(_checkpoint(tmp_path, PDD_FILE), {
            "pdd_step_indices": GRID32_BLOCKS8,
            "vsa_ref_keep_rate": 0.1
        },
                         attention_backend="VIDEO_SPARSE_ATTN_H3",
                         VSA_tile_size=128,
                         VSA_sparsity=0.9)
    assert args.pipeline_config.pdd_step_indices == tuple(GRID32_BLOCKS8)
    assert "trained with" not in caplog.text


def test_resolve_checkpoint_settings_repeated_call(tmp_path):
    args = _run_args(_checkpoint(tmp_path, PDD_FILE))
    args.pipeline_config.resolve_checkpoint_settings(args)
    assert (args.attention_backend, args.VSA_tile_size, args.VSA_sparsity) == ("VIDEO_SPARSE_ATTN_H3", 128, 0.9)
    assert args.pipeline_config.pdd_step_indices == tuple(GRID32_BLOCKS8)
    assert args.pipeline_config.vsa_ref_keep_rate == 0.1


@pytest.mark.parametrize("config_settings,run_settings,match", [
    ({}, {"attention_backend": "FLASH_ATTN"},
     "trained with attention_backend=VIDEO_SPARSE_ATTN_H3; this run requests FLASH_ATTN"),
    ({}, {"VSA_tile_size": 64}, "trained with VSA_tile_size=128; this run requests 64"),
    ({"pdd_step_indices": (0, 8, 16, 24, 32)}, {}, re.escape("this run sets [0, 8, 16, 24, 32]")),
], ids=["backend", "tile_size", "partition"])
def test_resolve_checkpoint_settings_explicit_training_fixed_conflict(tmp_path, config_settings, run_settings,
                                                                      match):
    with pytest.raises(ValueError, match=match):
        _run_args(_checkpoint(tmp_path, PDD_FILE), config_settings, **run_settings)


def test_resolve_checkpoint_settings_env_backend_conflict(tmp_path, env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("FLASH_ATTN"))
    with pytest.raises(ValueError, match="this run requests FLASH_ATTN"):
        _run_args(_checkpoint(tmp_path, PDD_FILE))


def test_resolve_checkpoint_settings_explicit_sparsity_override(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        args = _run_args(_checkpoint(tmp_path, PDD_FILE), VSA_sparsity=0.8)
    assert args.VSA_sparsity == 0.8
    assert "trained with VSA_sparsity=0.9; this run uses 0.8" in caplog.text


def test_resolve_checkpoint_settings_explicit_keep_rate_override(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        args = _run_args(_checkpoint(tmp_path, PDD_FILE), {"vsa_ref_keep_rate": 0.2})
    assert args.pipeline_config.vsa_ref_keep_rate == 0.2
    assert "trained with vsa_ref_keep_rate=0.1; this run uses 0.2" in caplog.text


@pytest.mark.parametrize("config_settings,run_settings,match", [
    ({}, {"VSA_sparsity": 1.0}, re.escape("VSA_sparsity must be in [0, 1), got 1.0")),
    ({"vsa_ref_keep_rate": 1.0}, {}, re.escape("vsa_ref_keep_rate must be in (0, 1), got 1.0")),
], ids=["sparsity", "keep_rate"])
def test_resolve_checkpoint_settings_tunable_out_of_range(tmp_path, config_settings, run_settings, match):
    with pytest.raises(ValueError, match=match):
        _run_args(_checkpoint(tmp_path, PDD_FILE), config_settings, **run_settings)


def test_resolve_checkpoint_settings_dmd_steps_on_pdd_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="dmd_denoising_steps must be unset"):
        _run_args(_checkpoint(tmp_path, PDD_FILE), {"dmd_denoising_steps": [999, 749, 500, 250]})


# ------------------------------------------------- resolve_checkpoint_settings: PDD file validation


def test_resolve_checkpoint_settings_missing_field(tmp_path):
    inference_file = {key: value for key, value in PDD_FILE.items() if key != "vsa_sparsity"}
    with pytest.raises(ValueError, match=re.escape("missing ['vsa_sparsity']")):
        _run_args(_checkpoint(tmp_path, inference_file))


@pytest.mark.parametrize("pin", ["", None, "9bfb6693", "MiniMaxAI/MiniMax-H3@9bfb6693", "hf://MiniMaxAI/MiniMax-H3"])
def test_resolve_checkpoint_settings_unparsable_base_model_revision(tmp_path, pin):
    with pytest.raises(ValueError, match="must be hf://<repo id>@<revision>"):
        _run_args(_checkpoint(tmp_path, {**PDD_FILE, "base_model_revision": pin}))


def test_resolve_checkpoint_settings_unknown_field(tmp_path):
    with pytest.raises(ValueError, match=re.escape("unknown ['dmd_denoising_steps']")):
        _run_args(_checkpoint(tmp_path, {**PDD_FILE, "dmd_denoising_steps": [999, 500]}))


@pytest.mark.parametrize("field,value", [
    ("schema_version", "fasth3-inference-contract-v2"),
    ("schema", "fasth3-inference-contract-v2"),
    ("model_type", "t2va"),
    ("transformer_component", "transformer"),
    ("conditioning", "noised_references"),
    ("guidance_scale", 2.0),
    ("attention_backend", "FLASH_ATTN"),
    ("vsa_ref_policy", "dense"),
    ("grid_max_t", 1.0),
])
def test_resolve_checkpoint_settings_fixed_field_other_value(tmp_path, field, value):
    with pytest.raises(ValueError, match=re.escape(f"{field}={value!r} is unsupported")):
        _run_args(_checkpoint(tmp_path, {**PDD_FILE, field: value}))


@pytest.mark.parametrize("field,value,match", [
    ("pdd_steps", 16, "pdd_steps=16 disagrees with transformer_ref/config.json pdd_steps=32"),
    ("video_scheduler_shift", 10.0,
     "video_scheduler_shift=10.0 disagrees with scheduler/scheduler_config.json shift=12.0"),
    ("audio_scheduler_shift", 4.0,
     "audio_scheduler_shift=4.0 disagrees with audio_scheduler/scheduler_config.json shift=3.0"),
    ("num_inference_steps", 9, "num_inference_steps=9 disagrees with the 8 blocks of pdd_step_indices"),
    ("transformer_forwards", 4, "transformer_forwards=4 disagrees with the 8 blocks of pdd_step_indices"),
])
def test_resolve_checkpoint_settings_duplicate_disagrees_with_owner(tmp_path, field, value, match):
    with pytest.raises(ValueError, match=match):
        _run_args(_checkpoint(tmp_path, {**PDD_FILE, field: value}))


@pytest.mark.parametrize("owner", ["transformer_ref/config.json", "audio_scheduler/scheduler_config.json"])
def test_resolve_checkpoint_settings_missing_owner_file(tmp_path, owner):
    model_path = _checkpoint(tmp_path, PDD_FILE)
    (tmp_path / owner).unlink()
    with pytest.raises(ValueError, match=f"has fastvideo_inference.json but no {owner}"):
        _run_args(model_path)


@pytest.mark.parametrize("field,value,match", [
    ("vsa_tile_size", 96, "vsa_tile_size=96 must be one of"),
    ("vsa_sparsity", "0.9", "vsa_sparsity must be in \\[0, 1\\), got '0.9'"),
    ("vsa_ref_keep_rate", 1.0, "vsa_ref_keep_rate must be in \\(0, 1\\), got 1.0"),
])
def test_resolve_checkpoint_settings_invalid_trained_vsa_value(tmp_path, field, value, match):
    with pytest.raises(ValueError, match=match):
        _run_args(_checkpoint(tmp_path, {**PDD_FILE, field: value}))


def test_resolve_checkpoint_settings_transformer_config_without_pdd_steps(tmp_path):
    with pytest.raises(ValueError, match="pdd_steps=32 disagrees with transformer_ref/config.json pdd_steps=None"):
        _run_args(_checkpoint(tmp_path, PDD_FILE, pdd_steps=None))


@pytest.mark.parametrize("indices", [
    [0, 8, 4, 12, 16, 20, 24, 28, 32],
    [1, 4, 8, 12, 16, 20, 24, 28, 32],
    [0, 4, 8, 12, 16, 20, 24, 28, 31],
    [0, 4.0, 8, 12, 16, 20, 24, 28, 32],
    [32],
], ids=["decreasing", "starts_after_0", "ends_before_pdd_steps", "float_node", "single_node"])
def test_resolve_checkpoint_settings_malformed_partition(tmp_path, indices):
    with pytest.raises(ValueError, match=re.escape(f"pdd_step_indices={indices!r} must increase strictly from 0 to "
                                                   "pdd_steps=32")):
        _run_args(_checkpoint(tmp_path, {**PDD_FILE, "pdd_step_indices": indices}))


# ------------------------------------------------- resolve_checkpoint_settings: checkpoints without PDD fields


@pytest.mark.parametrize("inference_file", [None, DMD_FILE], ids=["base", "dmd"])
def test_resolve_checkpoint_settings_non_pdd_checkpoint(tmp_path, inference_file):
    args = _run_args(_checkpoint(tmp_path, inference_file, pdd_steps=None))
    assert (args.attention_backend, args.VSA_tile_size, args.VSA_sparsity) == (None, 256, 0.0)
    config = args.pipeline_config
    assert (config.pdd_step_indices, config.vsa_ref_keep_rate, config.dmd_denoising_steps) == (None, None, None)


@pytest.mark.parametrize("inference_file", [None, DMD_FILE], ids=["base", "dmd"])
@pytest.mark.parametrize("config_settings", [{"pdd_step_indices": (0, 16, 32)}, {"vsa_ref_keep_rate": 0.1}],
                         ids=["partition", "keep_rate"])
def test_resolve_checkpoint_settings_pdd_setting_on_non_pdd_checkpoint(tmp_path, inference_file, config_settings):
    with pytest.raises(ValueError, match="applies only to FastH3 PDD checkpoints"):
        _run_args(_checkpoint(tmp_path, inference_file, pdd_steps=None), config_settings)


# ------------------------------------------------- apply_request_constraints: the step count of a request


def _pdd_config():
    return MiniMaxH3PipelineConfig(pdd_step_indices=tuple(GRID32_BLOCKS8))


@pytest.mark.parametrize("sampling", [{}, {"num_inference_steps": 8}])
def test_apply_request_constraints_pdd_unset_or_matching_steps(sampling):
    request = parse_config(GenerationRequest, {"prompt": "fox", "sampling": sampling})
    sampling_param = _pdd_config().apply_request_constraints(request, SamplingParam(num_inference_steps=50))
    assert sampling_param.num_inference_steps == 8


def test_apply_request_constraints_pdd_conflicting_steps():
    request = parse_config(GenerationRequest, {"prompt": "fox", "sampling": {"num_inference_steps": 50}})
    with pytest.raises(ValueError, match="runs exactly 8 transformer forwards.*num_inference_steps=50"):
        _pdd_config().apply_request_constraints(request, SamplingParam())


def test_apply_request_constraints_pdd_python_request_counts_schema_default_as_set():
    """A GenerationRequest built in Python carries the schema default 50 as a set value."""
    request = normalize_generation_request(GenerationRequest(prompt="fox"))
    with pytest.raises(ValueError, match="num_inference_steps=50. Pass num_inference_steps=8"):
        _pdd_config().apply_request_constraints(request, SamplingParam())


def test_apply_request_constraints_non_pdd_checkpoint_keeps_request_steps():
    request = parse_config(GenerationRequest, {"prompt": "fox", "sampling": {"num_inference_steps": 50}})
    sampling_param = SamplingParam(num_inference_steps=50)
    assert MiniMaxH3PipelineConfig().apply_request_constraints(request, sampling_param) is sampling_param
    assert sampling_param.num_inference_steps == 50


# ------------------------------------------------- worker pipeline: initialize_pipeline and the transformer check


def test_initialize_pipeline_pdd_checkpoint_keeps_resolved_settings(tmp_path):
    model_path = _checkpoint(tmp_path, PDD_FILE)
    args = _run_args(model_path)
    _pipeline(model_path).initialize_pipeline(args)
    config = args.pipeline_config
    assert config.dit_config.arch_config.pdd_steps == 32
    assert config.pdd_step_indices == tuple(GRID32_BLOCKS8)
    assert config.dmd_denoising_steps is None


def test_initialize_pipeline_dmd_checkpoint_applies_rungs(tmp_path):
    model_path = _checkpoint(tmp_path, DMD_FILE, pdd_steps=None)
    args = _run_args(model_path)
    _pipeline(model_path).initialize_pipeline(args)
    assert args.pipeline_config.dmd_denoising_steps == DMD_FILE["dmd_denoising_steps"]
    assert args.pipeline_config.pdd_step_indices is None


def test_check_pdd_transformer_widened_transformer_without_pdd_file(tmp_path):
    model_path = _checkpoint(tmp_path, None, pdd_steps=32)
    with pytest.raises(ValueError, match="sets pdd_steps, but .* has no PDD fastvideo_inference.json"):
        _pipeline(model_path)._check_pdd_transformer(_run_args(model_path))


def test_check_pdd_transformer_pdd_checkpoint_on_t2va_pipeline(tmp_path):
    model_path = _checkpoint(tmp_path, PDD_FILE)
    with pytest.raises(ValueError, match="select MiniMaxH3Ref2VAModularPipeline"):
        _pipeline(model_path, MiniMaxH3ModularPipeline)._check_pdd_transformer(_run_args(model_path))


def test_check_pdd_transformer_pdd_checkpoint_on_ref2va_pipeline(tmp_path):
    model_path = _checkpoint(tmp_path, PDD_FILE)
    _pipeline(model_path)._check_pdd_transformer(_run_args(model_path))


# ------------------------------------------------- worker pipeline: _load_config checks, before any component loads

_GATE = "transformer_blocks.0.attn.to_gate_compress.weight"


def _write_gates(transformer_dir: Path, *, indexed: bool) -> None:
    """Give a transformer checkpoint one VSA compression gate, listed in a shard index or in a safetensors file."""
    if indexed:
        (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {_GATE: "diffusion_pytorch_model-00001-of-00001.safetensors"}}))
    else:
        save_file({_GATE: torch.zeros(2, 2)}, str(transformer_dir / "diffusion_pytorch_model.safetensors"))


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


@pytest.mark.parametrize("indexed", [True, False], ids=["shard_index", "safetensors_header"])
@pytest.mark.parametrize("inference_file", [None, DMD_FILE], ids=["base", "dmd"])
def test_load_config_gated_non_pdd_transformer_requires_vsa(tmp_path, hub_snapshot, inference_file, indexed):
    """The gates are read from the resolved Hub snapshot. A PDD checkpoint's backend is checked by
    resolve_checkpoint_settings instead (test_resolve_checkpoint_settings_explicit_training_fixed_conflict)."""
    model_path = _checkpoint(tmp_path, inference_file, pdd_steps=None)
    _write_gates(tmp_path / "transformer_ref", indexed=indexed)
    pipeline = _pipeline("org/fasth3")
    pipeline.fastvideo_args = _run_args(model_path, attention_backend="FLASH_ATTN")
    with pytest.raises(ValueError, match="its transformer_ref carries VSA compression gates.*requests FLASH_ATTN"):
        pipeline._load_config(pipeline.model_path)
    assert hub_snapshot == ["org/fasth3"] and pipeline.model_path == model_path
    pipeline.fastvideo_args.attention_backend = "VIDEO_SPARSE_ATTN_H3"
    assert pipeline._load_config(pipeline.model_path) == {"_class_name": "MiniMaxH3ModularPipeline"}


@pytest.mark.parametrize("pdd_steps,gated,match", [
    (32, False, "has no PDD fastvideo_inference.json"),
    (None, True, "carries VSA compression gates"),
], ids=["widened_without_pdd_file", "gated_with_other_backend"])
def test_load_modules_transformer_check_before_loading_unless_supplied(tmp_path, monkeypatch, hub_snapshot,
                                                                       pdd_steps, gated, match):
    """A transformer that fails its check raises before any component loads; a transformer the caller already
    built is not rebuilt, so its checkpoint is not checked."""
    loads = []

    def base_load_modules(self, fastvideo_args, loaded_modules=None):
        self._load_config(self.model_path)
        loads.append(loaded_modules)
        return {}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", base_load_modules)
    model_path = _checkpoint(tmp_path, None, pdd_steps=pdd_steps)
    if gated:
        _write_gates(tmp_path / "transformer_ref", indexed=True)
    pipeline = _pipeline(model_path)
    args = _run_args(model_path, attention_backend="FLASH_ATTN", h3_sequential_load=False)
    pipeline.fastvideo_args = args
    with pytest.raises(ValueError, match=match):
        pipeline.load_modules(args)
    assert loads == []
    supplied = {"transformer": object()}
    pipeline.load_modules(args, supplied)
    assert loads == [supplied]


def test_checkpoint_facts_read_once_per_resolved_path(tmp_path, monkeypatch, hub_snapshot):
    """_load_config (gate check) and initialize_pipeline (DMD rungs) share one parse of the inference file."""
    from fastvideo.pipelines.basic.minimax_h3 import minimax_h3_pipeline

    parsed = []

    def recording_loads(text):
        parsed.append(json.loads(text))
        return parsed[-1]

    monkeypatch.setattr(minimax_h3_pipeline, "json", SimpleNamespace(loads=recording_loads))
    model_path = _checkpoint(tmp_path, DMD_FILE, pdd_steps=None)
    args = _run_args(model_path)
    pipeline = _pipeline("org/fasth3")
    pipeline.fastvideo_args = args
    pipeline._load_config(pipeline.model_path)
    pipeline.initialize_pipeline(args)
    assert args.pipeline_config.dmd_denoising_steps == DMD_FILE["dmd_denoising_steps"]
    assert parsed.count(DMD_FILE) == 1
