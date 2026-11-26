# SPDX-License-Identifier: Apache-2.0
from fastvideo.api.presets import InferencePreset, PresetStageSpec

BERNINI_R_1_3B = InferencePreset(
    name="bernini_r_1_3b",
    version=1,
    model_family="bernini",
    description="Bernini-R 1.3B video editing with source-aware conditioning and APG",
    stage_schemas=(PresetStageSpec(name="denoise",
                                   kind="denoising",
                                   allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"})), ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "negative_prompt": ""
    },
)

ALL_PRESETS = (BERNINI_R_1_3B, )
