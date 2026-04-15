# SPDX-License-Identifier: Apache-2.0
"""HunyuanGameCraft model family pipeline profiles."""
from fastvideo.api.profiles import PipelineProfile, ProfileStageSpec

_DENOISE_STAGE = ProfileStageSpec(
    name="denoise",
    kind="denoising",
    description="Action-controlled denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

GAMECRAFT_I2V = PipelineProfile(
    name="gamecraft_i2v",
    version="1",
    model_family="gamecraft",
    description="HunyuanGameCraft I2V at 704x1280",
    workload_type="i2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 33,
        "fps": 24,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
        "negative_prompt": "",
    },
)

ALL_PROFILES = (GAMECRAFT_I2V, )
