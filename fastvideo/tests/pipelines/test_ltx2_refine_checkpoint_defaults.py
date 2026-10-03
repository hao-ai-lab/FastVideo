# SPDX-License-Identifier: Apache-2.0
"""LTX-2 refine settings after the checkpoint's model_index.json defaults are applied at load time."""
from __future__ import annotations

from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.pipelines.basic.ltx2.ltx2_pipeline import LTX2Pipeline, _checkpoint_refine_values
from fastvideo.tests.api.config_snapshot import isolated_environment

LTX2 = "FastVideo/LTX2-Distilled-Diffusers"
CHECKPOINT_REFINE = {
    "fastvideo_refine_enabled": True,
    "fastvideo_refine_num_inference_steps": 2,
    "fastvideo_refine_guidance_scale": 4.0,
    "fastvideo_refine_add_noise": False,
    "fastvideo_refine_lora_path": "FastVideo/LTX2-Distilled-LoRA",
}


def _resolve(refine: dict | None = None):
    raw = {"model_path": LTX2}
    if refine is not None:
        raw["pipeline"] = {"ltx2": {"refine": refine}}
    with isolated_environment():
        return resolve_inference_config(raw)


def test_load_fills_unset_refine_settings_from_the_checkpoint_then_the_defaults(tmp_path):
    resolved = _resolve()
    pipeline = object.__new__(LTX2Pipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.resolved_config = resolved
    pipeline._required_config_modules = list(LTX2Pipeline._required_config_modules)
    model_index = {name: ["diffusers", "Model"] for name in pipeline._required_config_modules}
    pipeline._load_config = lambda model_path: {
        "_class_name": "LTX2Pipeline",
        "_diffusers_version": "0",
        "fastvideo_refine_num_inference_steps": 2,
        **model_index,
    }

    pipeline.load_modules(resolved, {name: object() for name in model_index})

    refine = pipeline.resolved_config.pipeline.ltx2.refine
    switches = (refine.enabled, refine.num_inference_steps, refine.guidance_scale, refine.add_noise)
    assert switches == (False, 2, 1.0, True)
    assert pipeline.resolved_config.provenance("pipeline.ltx2.refine.add_noise").source == "checkpoint:model_index.json"
    assert resolved.pipeline.ltx2.refine.num_inference_steps is None


def test_input_refine_settings_win_over_the_checkpoint():
    resolved = _resolve({"enabled": False, "num_inference_steps": 3, "guidance_scale": 2.0})

    values = _checkpoint_refine_values(resolved, "/unused", dict(CHECKPOINT_REFINE))

    assert values == {
        "pipeline.ltx2.refine.add_noise": False,
        "pipeline.ltx2.refine.lora_path": "FastVideo/LTX2-Distilled-LoRA",
    }
    assert _checkpoint_refine_values(_resolve(), "/unused", dict(CHECKPOINT_REFINE))["pipeline.ltx2.refine.enabled"]
