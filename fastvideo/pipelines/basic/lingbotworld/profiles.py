# SPDX-License-Identifier: Apache-2.0
"""LingBotWorld model family pipeline profiles."""
from fastvideo.api.profiles import PipelineProfile, ProfileStageSpec

_DENOISE_STAGE = ProfileStageSpec(
    name="denoise",
    kind="denoising",
    description="Dual-guidance denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "boundary_ratio",
    }),
)

LINGBOTWORLD_I2V = PipelineProfile(
    name="lingbotworld_i2v",
    version="1",
    model_family="lingbotworld",
    description="LingBot-World I2V with dual guidance",
    workload_type="i2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "guidance_scale": 5.0,
        "guidance_scale_2": 5.0,
        "num_inference_steps": 70,
        "fps": 16,
        "boundary_ratio": 0.947,
    },
)

ALL_PROFILES = (LINGBOTWORLD_I2V, )
