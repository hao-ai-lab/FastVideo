# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class Gen3C_Cosmos_7B_SamplingParam(SamplingParam):
    """Defaults for GEN3C (Cosmos-7B) camera-controlled video generation."""

    # Video parameters (720p, matching official GEN3C defaults)
    height: int = 720
    width: int = 1280
    num_frames: int = 121
    fps: int = 24

    # Denoising stage
    guidance_scale: float = 6.0
    negative_prompt: str = ""
    num_inference_steps: int = 50
