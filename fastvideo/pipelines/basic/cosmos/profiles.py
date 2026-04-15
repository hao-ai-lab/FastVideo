# SPDX-License-Identifier: Apache-2.0
"""Cosmos model family pipeline profiles.

Covers both Cosmos Predict2 and Cosmos Predict2.5, which share the
same pipeline directory but have distinct model families.
"""
from fastvideo.api.profiles import PipelineProfile, ProfileStageSpec

_DENOISE_STAGE = ProfileStageSpec(
    name="denoise",
    kind="denoising",
    description="Main denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

# -------------------------------------------------------------------
# Cosmos Predict2
# -------------------------------------------------------------------

COSMOS_PREDICT2_2B = PipelineProfile(
    name="cosmos_predict2_2b",
    version="1",
    model_family="cosmos",
    description="Cosmos Predict2 2B Video2World",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 93,
        "fps": 16,
        "guidance_scale": 7.0,
        "num_inference_steps": 35,
    },
)

# -------------------------------------------------------------------
# Cosmos Predict2.5
# -------------------------------------------------------------------

COSMOS25_PREDICT2_2B = PipelineProfile(
    name="cosmos25_predict2_2b",
    version="1",
    model_family="cosmos25",
    description="Cosmos Predict2.5 2B",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 77,
        "fps": 24,
        "guidance_scale": 7.0,
        "num_inference_steps": 35,
    },
)

ALL_PROFILES = (COSMOS_PREDICT2_2B, COSMOS25_PREDICT2_2B)
