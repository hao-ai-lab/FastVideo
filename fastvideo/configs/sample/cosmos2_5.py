# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class Cosmos_Predict2_5_2B_Diffusers_SamplingParam(SamplingParam):
    """Defaults for Cosmos 2.5 (Predict2.5) text-to-video diffusers-format model."""

    height: int = 480
    width: int = 832
    num_frames: int = 121
    fps: int = 24

    guidance_scale: float = 7.0
    # Official Cosmos2.5 sampling uses empty string as unconditional.
    negative_prompt: str = ""
    num_inference_steps: int = 35
