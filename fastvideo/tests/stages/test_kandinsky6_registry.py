# SPDX-License-Identifier: Apache-2.0
"""Registry routing and presets of the Kandinsky6 TI2VA checkpoints: base and pi-Flow distilled, Pro and Lite sizes.

Both official Diffusers repos declare the same model_index ``_class_name`` (``Kandinsky6TI2VAPipeline``), so they are
told apart by repo id or directory name, laid out like the LTX-2 distilled/base pair: the distilled entry is registered
first and the base detector excludes distilled names. Hub ids resolve by exact match (no network); local checkpoints are
model_index.json-only directories. The VSR ids are covered in ``kandinsky6_sr/test_kandinsky6_sr_registry_routing.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastvideo import registry
from fastvideo.api.compat import normalize_generation_request, request_to_sampling_param
from fastvideo.api.presets import get_preset, validate_preset_selection
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.kandinsky6 import Kandinsky6TI2VAConfig
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.pipelines.basic.kandinsky6.presets import (
    ALL_PRESETS,
    KANDINSKY6_TI2VA_DISTILLED,
    KANDINSKY6_TI2VA,
)

BASE_ID = "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"
OLD_BASE_ID = "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers"
DISTILLED_ID = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
LITE_ID = "kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers"
LITE_DISTILLED_ID = "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers"
BASE_PRESET = "kandinsky6_ti2va"
DISTILLED_PRESET = "kandinsky6_ti2va_distilled"


def _model_dir(root: Path, name: str) -> str:
    path = root / name
    path.mkdir(parents=True)
    model_index = {"_class_name": "Kandinsky6TI2VAPipeline", "_diffusers_version": "0.41.0.dev0"}
    (path / "model_index.json").write_text(json.dumps(model_index))
    return str(path)


def _matching_presets(text: str) -> set[str]:
    """Default presets of the Kandinsky6 registry entries whose detector claims ``text``."""
    entries = (registry._CONFIG_REGISTRY[model_id] for model_id, detector in registry._MODEL_NAME_DETECTORS
               if detector(text))
    return {entry.default_preset for entry in entries if entry.model_family in ("kandinsky6", "kandinsky6_sr")}


@pytest.mark.parametrize("repo_id,preset", [(BASE_ID, BASE_PRESET), (OLD_BASE_ID, BASE_PRESET),
                                            (DISTILLED_ID, DISTILLED_PRESET), (LITE_ID, BASE_PRESET),
                                            (LITE_DISTILLED_ID, DISTILLED_PRESET)])
def test_official_repo_ids_resolve_to_their_presets(repo_id, preset):
    assert repo_id in registry.get_registered_model_paths()
    assert registry.get_pipeline_config_cls_from_name(repo_id) is Kandinsky6TI2VAConfig
    assert registry.get_default_preset(repo_id) == preset
    assert registry.get_preset_selection(repo_id) == (preset, "kandinsky6")
    assert registry.get_model_family(repo_id) == "kandinsky6"
    info = registry._get_config_info(repo_id)
    assert info.pipeline_cls_name == "Kandinsky6TI2VAPipeline"
    assert info.workload_types == (WorkloadType.T2V, WorkloadType.I2V)


@pytest.mark.parametrize("repo_id", [BASE_ID, DISTILLED_ID, LITE_ID, LITE_DISTILLED_ID])
def test_get_model_info_resolves_both_repo_ids_to_the_ti2va_pipeline(monkeypatch, repo_id):
    model_index = {"_class_name": "Kandinsky6TI2VAPipeline", "_diffusers_version": "0.41.0.dev0"}
    monkeypatch.setattr(registry, "maybe_download_model_index", lambda *_args, **_kwargs: model_index)
    info = registry.get_model_info(repo_id)
    assert info.pipeline_cls.__name__ == "Kandinsky6TI2VAPipeline"
    assert info.pipeline_config_cls is Kandinsky6TI2VAConfig


def test_registered_model_listing_offers_both_repo_ids_for_text_and_image_workloads():
    listed = {model["id"]: model["workload_types"] for model in registry.get_registered_models_with_workloads()}
    assert listed[BASE_ID] == listed[DISTILLED_ID] == listed[LITE_ID] == listed[LITE_DISTILLED_ID] == ["t2v", "i2v"]


@pytest.mark.parametrize("name,preset", [
    ("Kandinsky-6.0-Pro-5s-Diffusers", BASE_PRESET),
    ("Kandinsky-6.0-Pro-sft-5s-Diffusers", BASE_PRESET),
    ("Kandinsky-6.0-Pro-distill-5s-Diffusers", DISTILLED_PRESET),
    ("Kandinsky-6.0-Lite-5s-Diffusers", BASE_PRESET),
    ("Kandinsky-6.0-Lite-distill-5s-Diffusers", DISTILLED_PRESET),
    ("kandinsky6-distilled-export", DISTILLED_PRESET),
    ("kandinsky6-export", BASE_PRESET),
    ("my_export", BASE_PRESET),
])
def test_local_directories_route_by_name(tmp_path, name, preset):
    path = _model_dir(tmp_path, name)
    assert registry.get_pipeline_config_cls_from_name(path) is Kandinsky6TI2VAConfig
    assert registry.get_preset_selection(path) == (preset, "kandinsky6")


def test_a_trailing_slash_does_not_hide_the_distilled_name(tmp_path):
    assert registry.get_default_preset(_model_dir(tmp_path, "kandinsky6-distilled-export") + "/") == DISTILLED_PRESET


def test_distill_in_a_parent_directory_does_not_make_a_base_checkpoint_distilled(tmp_path):
    path = _model_dir(tmp_path / "distill_experiments", "kandinsky6-export")
    assert registry.get_default_preset(path) == BASE_PRESET


@pytest.mark.parametrize("text,expected", [
    ("kandinsky-6.0-pro-distill-5s-diffusers", {DISTILLED_PRESET}),
    ("/models/kandinsky6-distilled-export", {DISTILLED_PRESET}),
    ("/models/kandinsky6-distilled-export/", {DISTILLED_PRESET}),
    ("kandinsky-6.0-pro-sft-5s-diffusers", {BASE_PRESET}),
    ("kandinsky-6.0-lite-5s-diffusers", {BASE_PRESET}),
    ("kandinsky-6.0-lite-distill-5s-diffusers", {DISTILLED_PRESET}),
    ("kandinsky6ti2vapipeline", {BASE_PRESET}),
    ("/distill_experiments/kandinsky6-export", {BASE_PRESET}),
])
def test_exactly_one_kandinsky6_detector_claims_a_name(text, expected):
    # routing must not depend on registration order alone: the base detector excludes distilled names
    assert _matching_presets(text) == expected


def test_ti2va_presets_are_registered():
    assert ALL_PRESETS == (KANDINSKY6_TI2VA, KANDINSKY6_TI2VA_DISTILLED)
    for preset in ALL_PRESETS:
        assert get_preset(preset.name, "kandinsky6") is preset


def test_base_preset_keeps_the_diffusers_defaults():
    defaults = get_preset(BASE_PRESET, "kandinsky6").defaults
    assert (defaults["num_inference_steps"], defaults["guidance_scale"]) == (50, 5.0)
    assert (defaults["height"], defaults["width"], defaults["num_frames"], defaults["fps"]) == (512, 768, 121, 24)


def _without_steps_and_guidance(defaults: dict) -> dict:
    return {key: value for key, value in defaults.items() if key not in ("num_inference_steps", "guidance_scale")}


def test_distilled_preset_differs_from_the_base_preset_only_in_steps_and_guidance():
    base = get_preset(BASE_PRESET, "kandinsky6")
    distilled = get_preset(DISTILLED_PRESET, "kandinsky6")
    assert (distilled.defaults["num_inference_steps"], distilled.defaults["guidance_scale"]) == (10, 1.0)
    assert _without_steps_and_guidance(distilled.defaults) == _without_steps_and_guidance(base.defaults)
    assert (distilled.model_family, distilled.workload_type) == (base.model_family, base.workload_type)
    assert distilled.stage_schemas == base.stage_schemas


@pytest.mark.parametrize("preset", [BASE_PRESET, DISTILLED_PRESET])
def test_denoise_stage_overrides_accept_steps_and_guidance(preset):
    overrides = {"denoise": {"num_inference_steps": 8, "guidance_scale": 1.0}}
    validate_preset_selection(preset, "kandinsky6", stage_overrides=overrides)


@pytest.mark.parametrize("repo_id,steps,guidance", [(BASE_ID, 50, 5.0), (DISTILLED_ID, 10, 1.0), (LITE_ID, 50, 5.0),
                                                     (LITE_DISTILLED_ID, 10, 1.0)])
def test_sampling_param_defaults_follow_the_repo_id(repo_id, steps, guidance):
    sampling_param = SamplingParam.from_pretrained(repo_id)
    assert (sampling_param.num_inference_steps, sampling_param.guidance_scale) == (steps, guidance)
    assert (sampling_param.height, sampling_param.width) == (512, 768)
    assert (sampling_param.num_frames, sampling_param.fps) == (121, 24)


def _request_sampling_param(model_path: str, **sampling) -> SamplingParam:
    request = normalize_generation_request({"prompt": "a chef chops vegetables", "sampling": sampling})
    return request_to_sampling_param(request, model_path=model_path)


def test_explicit_request_values_win_over_the_distilled_preset():
    sampling_param = _request_sampling_param(DISTILLED_ID, num_inference_steps=8)
    assert (sampling_param.num_inference_steps, sampling_param.guidance_scale) == (8, 1.0)


def test_explicit_request_values_win_over_the_base_preset():
    sampling_param = _request_sampling_param(BASE_ID, num_inference_steps=30, guidance_scale=3.0)
    assert (sampling_param.num_inference_steps, sampling_param.guidance_scale) == (30, 3.0)
    sampling_param = _request_sampling_param(BASE_ID, guidance_scale=3.0)
    assert (sampling_param.num_inference_steps, sampling_param.guidance_scale) == (50, 3.0)
