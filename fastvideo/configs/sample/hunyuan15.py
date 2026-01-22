# SPDX-License-Identifier: Apache-2.0
import numpy as np
from dataclasses import dataclass, field

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class Hunyuan15_480P_SamplingParam(SamplingParam):
    num_inference_steps: int = 50

    num_frames: int = 121
    height: int = 480
    width: int = 848
    fps: int = 24

    guidance_scale: float = 6.0
    prompt_attention_mask: list = field(default_factory=list)
    negative_attention_mask: list = field(default_factory=list)
    sigmas: list[float] | None = field(
        default_factory=lambda: list(np.linspace(1.0, 0.0, 50 + 1)[:-1]))

    negative_prompt: str = ""
    
    def __post_init__(self):
        self.sigmas = list(
            np.linspace(1.0, 0.0, self.num_inference_steps + 1)[:-1]
        )

@dataclass
class Hunyuan15_720P_SamplingParam(Hunyuan15_480P_SamplingParam):
    height: int = 720
    width: int = 1280
