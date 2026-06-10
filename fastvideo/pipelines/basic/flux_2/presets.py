# SPDX-License-Identifier: Apache-2.0
"""Flux2 model family pipeline presets.

Each preset is a named inference preset that declares the user-facing
stage topology, default sampling values, and which per-stage overrides
are allowed.  Presets are registered explicitly from
:func:`fastvideo.registry._register_presets`.
"""
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

FLUX2_DEV = InferencePreset(
    name="flux2_dev",
    version=1,
    model_family="flux2",
    description="Flux2 full T2I with embedded guidance",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "seed": 0,
        "guidance_scale": 4.0,
        "num_inference_steps": 50,
    },
)

FLUX2_KLEIN_4B = InferencePreset(
    name="flux2_klein_4b",
    version=1,
    model_family="flux2",
    description="Flux2 Klein 4B (distilled, 4-step, no guidance)",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "seed": 0,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
    },
)

FLUX2_KLEIN_9B = InferencePreset(
    name="flux2_klein_9b",
    version=1,
    model_family="flux2",
    description="Flux2 Klein 9B (distilled, 4-step, no guidance)",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "seed": 0,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
    },
)

ALL_PRESETS = (FLUX2_DEV, FLUX2_KLEIN_4B, FLUX2_KLEIN_9B)
