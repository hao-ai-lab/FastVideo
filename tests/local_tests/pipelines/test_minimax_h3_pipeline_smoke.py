# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 public registration, modular manifest, and preset smoke tests."""

from __future__ import annotations

import json
from pathlib import Path


def _write_modular_checkpoint(model_dir: Path) -> None:
    component_types = {
        "text_encoder": ("transformers", "Qwen3VLForConditionalGeneration"),
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
        "processor": ("transformers", "Qwen3VLProcessor"),
        "vae": ("diffusers", "AutoencoderKLMiniMaxH3"),
        "audio_vae": ("diffusers", "AutoencoderKLMiniMaxH3Audio"),
        "transformer": ("diffusers", "MiniMaxH3Transformer3DModel"),
        "transformer_ref": ("diffusers", "MiniMaxH3Transformer3DModel"),
        "scheduler": ("diffusers", "MiniMaxH3Scheduler"),
        "audio_scheduler": ("diffusers", "MiniMaxH3Scheduler"),
    }
    components = {
        name: [*component_type, {"type_hint": list(component_type), "subfolder": name}]
        for name, component_type in component_types.items()
    }
    model_dir.mkdir()
    for component in components:
        (model_dir / component).mkdir()
    manifest = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "_diffusers_version": "0.36.0.dev0",
        "_blocks_class_name": "MiniMaxH3Blocks",
        **components,
    }
    (model_dir / "modular_model_index.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_minimax_h3_registry_resolves_both_public_pipelines(tmp_path: Path) -> None:
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
        MiniMaxH3ModularPipeline,
        MiniMaxH3Ref2VAModularPipeline,
    )
    from fastvideo.registry import get_model_info

    model_dir = tmp_path / "checkpoint-with-modular-manifest"
    _write_modular_checkpoint(model_dir)

    default_info = get_model_info(str(model_dir), workload_type=WorkloadType.I2V)
    assert default_info.pipeline_cls is MiniMaxH3ModularPipeline
    assert default_info.pipeline_config_cls is MiniMaxH3PipelineConfig

    ref_info = get_model_info(
        str(model_dir),
        workload_type=WorkloadType.I2V,
        override_pipeline_cls_name="MiniMaxH3Ref2VAModularPipeline",
    )
    assert ref_info.pipeline_cls is MiniMaxH3Ref2VAModularPipeline
    assert ref_info.pipeline_config_cls is MiniMaxH3PipelineConfig
    assert MiniMaxH3Ref2VAModularPipeline._extra_config_module_map == {"transformer": "transformer_ref"}


def test_minimax_h3_presets_and_guidance_distilled_defaults(tmp_path: Path) -> None:
    from fastvideo.api.presets import get_preset
    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.registry import get_preset_selection

    model_dir = tmp_path / "minimax_h3-local"
    _write_modular_checkpoint(model_dir)

    preset_name, family = get_preset_selection(str(model_dir))
    assert (preset_name, family) == ("minimax_h3_t2va", "minimax_h3")
    preset = get_preset(preset_name, family)
    assert preset.defaults == {
        "fps": 24,
        "guidance_scale": 1.0,
        "batch_cfg": False,
        "negative_prompt": "",
        "num_inference_steps": 50,
        "seed": 0,
        "height": 768,
        "width": 1344,
        "num_frames": 124,
    }

    sampling = SamplingParam.from_pretrained(str(model_dir))
    assert sampling.negative_prompt == ""
    assert sampling.guidance_scale == 1.0
    assert sampling.batch_cfg is False

    fl2va = get_preset("minimax_h3_fl2va", family)
    ref2va = get_preset("minimax_h3_ref2va", family)
    assert fl2va.defaults["num_frames"] == 192
    assert ref2va.defaults["num_frames"] == 124
