# SPDX-License-Identifier: Apache-2.0
"""Pipeline profiles for Stable Diffusion 3.5 model family.

Each profile defines default sampling parameters that differ from the
base ``SamplingParam`` defaults.
"""

from fastvideo.configs.sample.profiles import (
    PipelineProfile,
    register_profile,
)

SD35_MEDIUM = PipelineProfile(
    name="sd35_medium",
    defaults={
        "negative_prompt": "",
        "num_videos_per_prompt": 1,
        "seed": 0,
        "num_frames": 1,
        "height": 512,
        "width": 512,
        "fps": 1,
        "num_inference_steps": 28,
        "guidance_scale": 6.0,
    },
)

register_profile(SD35_MEDIUM)
