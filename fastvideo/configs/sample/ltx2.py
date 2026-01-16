# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class LTX2SamplingParam(SamplingParam):
    """Default sampling parameters for LTX-2 distilled T2V.
    """

    seed: int = 10
    num_frames: int = 121
    height: int = 1024
    width: int = 1536
    fps: int = 24
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    # No default negative_prompt for distilled models
    negative_prompt: str = ""
