# SPDX-License-Identifier: Apache-2.0
"""LTX2 model family pipeline profiles."""
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

LTX2_BASE = PipelineProfile(
    name="ltx2_base",
    version="1",
    model_family="ltx2",
    description="LTX-2 base at 512x768",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 3.0,
        "num_inference_steps": 40,
    },
)

LTX2_DISTILLED = PipelineProfile(
    name="ltx2_distilled",
    version="1",
    model_family="ltx2",
    description="LTX-2 distilled at 1024x1536",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 1024,
        "width": 1536,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
    },
)

ALL_PROFILES = (LTX2_BASE, LTX2_DISTILLED)
