# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.sample.base import SamplingParam
import numpy as np


@dataclass
class HyWorld_SamplingParam(SamplingParam):
    num_inference_steps: int = 50

    num_frames: int = 125
    height: int = 480
    width: int = 832
    fps: int = 24

    # Camera trajectory: pose string (e.g., 'w-31' means generating [1 + 31] latents) or JSON file path
    pose: str = 'w-31'

    guidance_scale: float = 6.0
    prompt_attention_mask: list = field(default_factory=list)
    negative_attention_mask: list = field(default_factory=list)
    sigmas: list[float] | None = field(
        default_factory=lambda: list(np.linspace(1.0, 0.0, 50 + 1)[:-1]))

    negative_prompt: str = ""
