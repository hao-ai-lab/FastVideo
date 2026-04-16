# SPDX-License-Identifier: Apache-2.0
"""TurboDiffusion model family pipeline presets."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Fast few-step denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

TURBO_T2V_1_3B = InferencePreset(
    name="turbo_t2v_1_3b",
    version=1,
    model_family="turbodiffusion",
    description="TurboWan 2.1 T2V 1.3B (4-step)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "negative_prompt": None,
    },
)

TURBO_T2V_14B = InferencePreset(
    name="turbo_t2v_14b",
    version=1,
    model_family="turbodiffusion",
    description="TurboWan 2.1 T2V 14B (4-step)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "negative_prompt": None,
    },
)

TURBO_I2V_A14B = InferencePreset(
    name="turbo_i2v_a14b",
    version=1,
    model_family="turbodiffusion",
    description="TurboWan 2.2 I2V A14B (4-step)",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "negative_prompt": None,
    },
)

ALL_PRESETS = (
    TURBO_T2V_1_3B,
    TURBO_T2V_14B,
    TURBO_I2V_A14B,
)
