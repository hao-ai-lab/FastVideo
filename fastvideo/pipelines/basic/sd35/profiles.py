# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion 3.5 model family pipeline profiles."""
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

SD35_MEDIUM = PipelineProfile(
    name="sd35_medium",
    version="1",
    model_family="sd35",
    description="Stable Diffusion 3.5 Medium (text-to-image)",
    workload_type="t2i",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 512,
        "num_frames": 1,
        "fps": 1,
        "guidance_scale": 6.0,
        "num_inference_steps": 28,
    },
)

ALL_PROFILES = (SD35_MEDIUM, )
