# SPDX-License-Identifier: Apache-2.0
"""LTX-2 refine settings after resolution fills them from the checkpoint's model_index.json."""
from __future__ import annotations

import json

from fastvideo.api.checkpoint_defaults import ltx2_refine_checkpoint_step
from fastvideo.api.inference_resolution import fill_runtime_defaults, resolve_inference_config
from fastvideo.api.resolution import resolve_generator_config
from fastvideo.pipelines.basic.ltx2.pipeline_configs import LTX2T2VConfig
from fastvideo.tests.api.config_snapshot import isolated_environment

LTX2 = "FastVideo/LTX2-Distilled-Diffusers"
CHECKPOINT_REFINE = {
    "fastvideo_refine_enabled": True,
    "fastvideo_refine_num_inference_steps": 2,
    "fastvideo_refine_guidance_scale": 4.0,
    "fastvideo_refine_add_noise": False,
    "fastvideo_refine_lora_path": "FastVideo/LTX2-Distilled-LoRA",
}
REFINE_SWITCHES = (
    "pipeline.ltx2.refine.enabled",
    "pipeline.ltx2.refine.num_inference_steps",
    "pipeline.ltx2.refine.guidance_scale",
    "pipeline.ltx2.refine.add_noise",
)


def _checkpoint(tmp_path, **manifest_entries) -> str:
    """A local LTX-2 checkpoint directory whose model_index.json holds ``manifest_entries``."""
    (tmp_path / "transformer").mkdir()
    (tmp_path / "model_index.json").write_text(
        json.dumps({
            "_class_name": "LTX2Pipeline",
            "_diffusers_version": "0",
            "transformer": ["diffusers", "Model"],
            **manifest_entries
        }))
    return str(tmp_path)


def _resolve_local(model_path: str, refine: dict | None = None, upsampler_weights: str | None = None):
    """Resolve a local LTX-2 checkpoint through the checkpoint step and the runtime defaults only."""
    raw: dict = {"model_path": model_path, "pipeline": {"components": {"upsampler_weights": upsampler_weights}}}
    if refine is not None:
        raw["pipeline"]["ltx2"] = {"refine": refine}
    return resolve_generator_config(raw, (ltx2_refine_checkpoint_step(LTX2T2VConfig()), fill_runtime_defaults))


def test_checkpoint_fills_unset_refine_settings_then_the_defaults_fill_the_rest(tmp_path):
    model_path = _checkpoint(tmp_path, fastvideo_refine_num_inference_steps=2)
    (tmp_path / "spatial_upscaler").mkdir()

    resolved = _resolve_local(model_path)

    refine = resolved.pipeline.ltx2.refine
    switches = (refine.enabled, refine.num_inference_steps, refine.guidance_scale, refine.add_noise)
    assert switches == (False, 2, 1.0, True)
    assert resolved.provenance("pipeline.ltx2.refine.num_inference_steps").source == "fill_ltx2_refine_from_checkpoint"
    assert resolved.provenance("pipeline.ltx2.refine.add_noise").source == "fill_runtime_defaults"
    assert resolved.pipeline.components.upsampler_weights == str(tmp_path / "spatial_upscaler")


def test_input_refine_settings_win_over_the_checkpoint(tmp_path):
    model_path = _checkpoint(tmp_path, **CHECKPOINT_REFINE)

    resolved = _resolve_local(model_path, {"enabled": False, "num_inference_steps": 3, "guidance_scale": 2.0})

    checkpoint_decisions = [
        values for source, values in resolved.decisions if source == "fill_ltx2_refine_from_checkpoint"
    ]
    assert checkpoint_decisions == [{
        "pipeline.ltx2.refine.add_noise": False,
        "pipeline.ltx2.refine.lora_path": "FastVideo/LTX2-Distilled-LoRA",
    }]
    assert resolved.pipeline.ltx2.refine.enabled is False
    assert _resolve_local(model_path).pipeline.ltx2.refine.enabled is True


def test_relative_refine_paths_resolve_inside_the_checkpoint(tmp_path):
    (tmp_path / "transformer_refine").mkdir()
    model_path = _checkpoint(tmp_path,
                             fastvideo_refine_transformer_path="transformer_refine",
                             fastvideo_refine_noise_path="/weights/noise.safetensors",
                             spatial_upsampler=["diffusers", "LTX2LatentUpsampler"])
    (tmp_path / "spatial_upsampler").mkdir()

    resolved = _resolve_local(model_path)

    assert resolved.pipeline.ltx2.refine.transformer_path == str(tmp_path / "transformer_refine")
    assert resolved.pipeline.ltx2.refine.noise_path == "/weights/noise.safetensors"
    assert resolved.pipeline.components.upsampler_weights == str(tmp_path / "spatial_upsampler")


def test_step_skips_other_models_and_blocked_downloads(tmp_path):
    model_path = _checkpoint(tmp_path, **CHECKPOINT_REFINE)
    other_model = resolve_generator_config({"model_path": model_path},
                                           (ltx2_refine_checkpoint_step(object()), fill_runtime_defaults))
    assert other_model.pipeline.ltx2.refine.enabled is False
    assert other_model.pipeline.ltx2.refine.lora_path is None

    with isolated_environment():
        hub_model = resolve_inference_config({"model_path": LTX2})

    assert [source for source, _ in hub_model.decisions if source == "fill_ltx2_refine_from_checkpoint"] == []
    assert tuple(hub_model.provenance(path).value for path in REFINE_SWITCHES) == (False, 3, 1.0, True)
    assert hub_model.provenance("pipeline.ltx2.refine.enabled").source == "fill_runtime_defaults"
