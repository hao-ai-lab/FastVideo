# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion 3.5 model family pipeline presets."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Main denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

SD35_MEDIUM = InferencePreset(
    name="sd35_medium",
    version=1,
    model_family="sd35",
    description="Stable Diffusion 3.5 Medium (text-to-image)",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 512,
        "num_frames": 1,
        "fps": 1,
        "seed": 0,
        "guidance_scale": 6.0,
        "num_inference_steps": 28,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (SD35_MEDIUM, )
