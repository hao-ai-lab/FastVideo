# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 defaults shared by text and reference image generation."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

QWEN_IMAGE21 = InferencePreset(
    name="qwen_image21",
    version=1,
    model_family="qwen_image21",
    description="Qwen-Image-2.1 image generation, editing and ordered references",
    workload_type="t2i",
    stage_schemas=(PresetStageSpec(name="denoise",
                                   kind="denoising",
                                   description="Qwen-Image-2.1 Euler flow steps",
                                   allowed_overrides=frozenset({"num_inference_steps", "true_cfg_scale"})), ),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "num_videos_per_prompt": 1,
        "fps": 1,
        "seed": 42,
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "true_cfg_scale": 1.0,
        "negative_prompt": None,
        "reference_resolution": 1024,
        "use_kv_cache": True,
    },
)

QWEN_IMAGE21_EDIT = InferencePreset(
    name="qwen_image21_edit",
    version=1,
    model_family="qwen_image21",
    description="Qwen-Image-2.1 image edits and reference-to-image generation",
    workload_type="i2i",
    stage_schemas=QWEN_IMAGE21.stage_schemas,
    defaults=dict(QWEN_IMAGE21.defaults),
)

ALL_PRESETS = (QWEN_IMAGE21, QWEN_IMAGE21_EDIT)
