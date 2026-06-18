# SPDX-License-Identifier: Apache-2.0
"""Ovis-Image model family pipeline presets."""
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

# Defaults mirror the official OvisImagePipeline example (1024², 50 steps,
# guidance 5.0); see examples/inference/basic/basic_ovis_image.py.
OVIS_IMAGE_7B = InferencePreset(
    name="ovis_image_7b",
    version=1,
    model_family="ovis_image",
    description="Ovis-Image-7B (text-to-image, optimized for text rendering)",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "seed": 0,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (OVIS_IMAGE_7B, )
