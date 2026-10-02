# SPDX-License-Identifier: Apache-2.0
"""Cosmos Predict pipeline presets."""
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

_COSMOS_PREDICT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, "
    "static with no motion, motion blur, over-saturation, shaky "
    "footage, low resolution, grainy texture, pixelated images, "
    "poorly lit areas, underexposed and overexposed scenes, poor "
    "color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, artifacting, color banding, "
    "unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual "
    "noise, and flickering. Overall, the video is of poor quality."
)

COSMOS_PREDICT_7B = InferencePreset(
    name="cosmos_predict_preset",
    version=1,
    model_family="cosmos_predict",
    description="Cosmos 1.0 Prompt2World 7B Video",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 0,
        "height": 704,
        "width": 1280,
        "num_frames": 93,
        "fps": 24,
        "guidance_scale": 7.0,
        "num_inference_steps": 35,
        "negative_prompt": _COSMOS_PREDICT_NEGATIVE_PROMPT,
    },
)

COSMOS_PREDICT_14B = InferencePreset(
    name="cosmos_predict_14b_preset",
    version=1,
    model_family="cosmos_predict",
    description="Cosmos 1.0 Prompt2World 14B Video",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 0,
        "height": 704,
        "width": 1280,
        "num_frames": 93,
        "fps": 24,
        "guidance_scale": 7.0,
        "num_inference_steps": 35,
        "negative_prompt": _COSMOS_PREDICT_NEGATIVE_PROMPT,
    },
)

ALL_PRESETS = (COSMOS_PREDICT_7B, COSMOS_PREDICT_14B)
