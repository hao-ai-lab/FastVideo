# SPDX-License-Identifier: Apache-2.0
"""Registry routing: SR bundles (both official VSR repo ids and local copies) resolve to the SR config, Kandinsky6 TI2VA
paths keep resolving to Kandinsky6."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from fastvideo import registry
from fastvideo.api.presets import get_preset, validate_preset_selection
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.kandinsky6 import Kandinsky6TI2VAConfig
from fastvideo.configs.pipelines.kandinsky6_sr import Kandinsky6SRPipelineConfig
from fastvideo.configs.pipelines.kandinsky6_sr_options import Kandinsky6SROptions, SR_OPTION_FIELDS
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


def _model_dir(root: Path, name: str, class_name: str) -> str:
    path = root / name
    path.mkdir(parents=True)
    (path / "model_index.json").write_text(json.dumps({"_class_name": class_name, "_diffusers_version": "0.35.0"}))
    return str(path)


def _config_cls(path: str):
    return registry._get_config_info(path).pipeline_config_cls


SR_FLOW_ID = "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers"
SR_DISTILLED_ID = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"
_PRESET = {SR_FLOW_ID: "kandinsky6_sr", SR_DISTILLED_ID: "kandinsky6_sr_distilled"}


@pytest.mark.parametrize("repo_id", [SR_FLOW_ID, SR_DISTILLED_ID])
def test_official_vsr_repo_ids_resolve_to_the_sr_config(repo_id):
    # exact registry match: no model_index fetch
    assert repo_id in registry.get_registered_model_paths()
    assert registry.get_pipeline_config_cls_from_name(repo_id) is Kandinsky6SRPipelineConfig
    assert registry.get_preset_selection(repo_id) == (_PRESET[repo_id], "kandinsky6_sr")
    info = registry._get_config_info(repo_id)
    assert info.pipeline_cls_name == "Kandinsky6SRPipeline"
    assert info.workload_types == ()


@pytest.mark.parametrize("repo_id", [SR_FLOW_ID, SR_DISTILLED_ID])
def test_get_model_info_resolves_both_vsr_repo_ids_to_the_sr_pipeline(monkeypatch, repo_id):
    model_index = {"_class_name": "Kandinsky6SRPipeline", "_diffusers_version": "0.39.0"}
    monkeypatch.setattr(registry, "maybe_download_model_index", lambda *_args, **_kwargs: model_index)
    info = registry.get_model_info(repo_id)
    assert info.pipeline_cls.__name__ == "Kandinsky6SRPipeline"
    assert info.pipeline_config_cls is Kandinsky6SRPipelineConfig


@pytest.mark.parametrize("name", ["Kandinsky-6.0-VSR-5s-Diffusers", "Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"])
def test_local_copies_of_the_official_vsr_repos_resolve_to_the_sr_config(tmp_path, name):
    # the "distill" marker of the distilled VSR repo must not route it to the distilled TI2VA entry
    path = _model_dir(tmp_path, name, "Kandinsky6SRPipeline")
    assert _config_cls(path) is Kandinsky6SRPipelineConfig
    assert registry.get_preset_selection(path) == (_PRESET[f"kandinskylab/{name}"], "kandinsky6_sr")


@pytest.mark.parametrize("name,class_name", [
    ("Kandinsky-6.0-Pro-5s-Diffusers", "Kandinsky6TI2VAPipeline"),
    ("Kandinsky-6.0-Pro-sft-5s-Diffusers", "Kandinsky6TI2VAPipeline"),
    ("Kandinsky-6.0-Pro-distill-5s-Diffusers", "Kandinsky6TI2VAPipeline"),
    ("kandinsky6-t2va", "Kandinsky6TI2VAPipeline"),
    ("my_kandinsky6_export", "Kandinsky6TI2VAPipeline"),
    ("kandinsky-6-export", "SomeOtherPipeline"),
])
def test_kandinsky6_ti2va_paths_still_resolve_to_kandinsky6(tmp_path, name, class_name):
    assert _config_cls(_model_dir(tmp_path, name, class_name)) is Kandinsky6TI2VAConfig


def test_a_parent_directory_called_sr_does_not_turn_a_ti2va_checkpoint_into_vsr(tmp_path):
    path = _model_dir(tmp_path / "sr" / "vsr_experiments", "kandinsky6-export", "Kandinsky6TI2VAPipeline")
    assert _config_cls(path) is Kandinsky6TI2VAConfig


@pytest.mark.parametrize("name", [
    "Kandinsky-6.0-VSR-distilled2steps-5s",
    "kandinsky6_sr",
    "kandinsky6-sr-export",
    "Kandinsky6SR",
    "kandinsky-6-super-res",
])
def test_sr_bundle_names_resolve_to_the_sr_config(tmp_path, name):
    # the model_index class name of an SR bundle is Kandinsky6SRPipeline; the names are also tried on their own
    assert _config_cls(_model_dir(tmp_path, name, "Kandinsky6SRPipeline")) is Kandinsky6SRPipelineConfig


def test_sr_bundle_resolves_from_the_model_index_class_name_alone(tmp_path):
    assert _config_cls(_model_dir(tmp_path, "vsr_bundle_v1", "Kandinsky6SRPipeline")) is Kandinsky6SRPipelineConfig


def test_the_ti2va_detector_never_matches_an_sr_bundle():
    # routing must not depend on registration order alone: exactly one config matches SR inputs
    sr_inputs = [
        "kandinsky-6.0-vsr-5s-diffusers", "kandinsky-6.0-vsr-distilled2steps-5s-diffusers",
        "kandinsky-6.0-vsr-distilled2steps-5s", "kandinsky6srpipeline", "kandinsky6_sr", "kandinsky-6-sr"
    ]
    for text in sr_inputs:
        matching = {registry._CONFIG_REGISTRY[model_id].pipeline_config_cls
                    for model_id, detector in registry._MODEL_NAME_DETECTORS if detector(text)}
        assert matching == {Kandinsky6SRPipelineConfig}, (text, matching)
    ti2va_inputs = [
        "kandinsky-6.0-pro-sft-5s-diffusers", "kandinsky-6.0-pro-distill-5s-diffusers", "kandinsky6-t2va",
        "kandinsky6ti2vapipeline", "kandinsky-6-export"
    ]
    for text in ti2va_inputs:
        matching = {registry._CONFIG_REGISTRY[model_id].pipeline_config_cls
                    for model_id, detector in registry._MODEL_NAME_DETECTORS if detector(text)}
        assert matching == {Kandinsky6TI2VAConfig}, (text, matching)


def test_sr_config_entry_metadata():
    info = registry._get_config_info(_registered_sr_path())
    assert info.pipeline_cls_name == "Kandinsky6SRPipeline"
    assert info.workload_types == ()  # no V2V WorkloadType: not exposed as a workload option
    assert info.default_preset == "kandinsky6_sr" and info.model_family == "kandinsky6_sr"


def _registered_sr_path() -> str:
    import tempfile

    return _model_dir(Path(tempfile.mkdtemp()), "kandinsky6_sr", "Kandinsky6SRPipeline")


def test_pipeline_class_is_discovered_under_the_model_index_name(distilled_bundle):
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.pipeline_registry import PipelineType, get_pipeline_registry

    index_name = json.loads((distilled_bundle / "model_index.json").read_text())["_class_name"]
    cls = get_pipeline_registry(PipelineType.BASIC).resolve_pipeline_cls(index_name, PipelineType.BASIC,
                                                                        WorkloadType.T2V)
    assert cls.__name__ == "Kandinsky6SRPipeline" == index_name
    assert cls._required_config_modules == ["transformer", "vae", "latent_upscaler", "scheduler"]


@pytest.mark.parametrize("name, steps", [("Kandinsky-6.0-VSR-5s-Diffusers", 4),
                                         ("Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", 2)])
def test_preset_defaults_reach_the_sampling_param(tmp_path, name, steps):
    preset = get_preset("kandinsky6_sr", "kandinsky6_sr")
    assert preset.stage_schemas[0].kind == "super_resolution"
    sp = SamplingParam.from_pretrained(_model_dir(tmp_path, name, "Kandinsky6SRPipeline"))
    assert sp.num_inference_steps == steps
    assert sp.seed == 42
    assert Kandinsky6SROptions.from_extra(preset.defaults) == Kandinsky6SROptions()
    assert not any(hasattr(sp, key) for key in SR_OPTION_FIELDS)
    assert sp.guidance_scale == 1.0 and sp.return_frames is False and sp.save_video is True


def test_stage_overrides_accept_only_the_sr_knobs():
    validate_preset_selection("kandinsky6_sr", "kandinsky6_sr", stage_overrides={"sr": {"num_inference_steps": 3}})
    from fastvideo.api.errors import ConfigValidationError

    with pytest.raises(ConfigValidationError):
        validate_preset_selection("kandinsky6_sr", "kandinsky6_sr", stage_overrides={"sr": {"seed": 1}})
    with pytest.raises(ConfigValidationError):
        validate_preset_selection("kandinsky6_sr", "kandinsky6_sr", stage_overrides={"denoise": {}})


def test_sr_options_are_not_global_sampling_or_batch_fields():
    for cls in (SamplingParam, ForwardBatch):
        assert not set(SR_OPTION_FIELDS).intersection(f.name for f in fields(cls))
    from fastvideo.utils import shallow_asdict

    ForwardBatch(**shallow_asdict(SamplingParam()), eta=0.0, n_tokens=1, VSA_sparsity=0.0)


@pytest.mark.parametrize("section", ["extensions", "stage_overrides"])
def test_request_extensions_reach_sr_options(tmp_path, section):
    from fastvideo.api.compat import normalize_generation_request, request_to_batch_extra, request_to_sampling_param

    model_path = _model_dir(tmp_path, "kandinsky6_sr", "Kandinsky6SRPipeline")
    overrides = {"sr_resolution_scale": 4, "sr_target_resolution": "fullhd"}
    request = normalize_generation_request({
        "inputs": {"video_path": "clip.mp4"},
        section: overrides if section == "extensions" else {"sr": overrides},
    })
    sp = request_to_sampling_param(request, model_path=model_path)
    options = Kandinsky6SROptions.from_extra(request_to_batch_extra(request))
    assert sp.video_path == "clip.mp4"
    assert options.sr_resolution_scale == 4 and options.sr_target_resolution == "fullhd"
    assert options.sr_tiles_batch_size == 1 and options.sr_tile_min_overlap == 0.20
    assert sp.seed == 42 and sp.num_inference_steps == 4


def test_generator_hooks_are_class_attributes_not_dataclass_fields():
    assert Kandinsky6SRPipelineConfig.prompt_optional is True
    assert Kandinsky6SRPipelineConfig.output_shape_from_input is True
    assert not getattr(Kandinsky6TI2VAConfig, "prompt_optional", False)
    names = {f.name for f in fields(Kandinsky6SRPipelineConfig)}
    assert "prompt_optional" not in names and "output_shape_from_input" not in names  # no inventory entries needed


def test_pipeline_config_needs_no_text_encoder():
    config = Kandinsky6SRPipelineConfig()
    config.check_pipeline_config()  # equal-length (empty) encoder / precision / preprocess lists, vae_sp off
    assert config.text_encoder_configs == () and config.dit_precision == "bf16"
    assert config.vae_config.load_encoder and config.vae_config.load_decoder
