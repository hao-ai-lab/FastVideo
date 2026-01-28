# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class FluxSamplingParam(SamplingParam):
    """Default sampling parameters for Flux T2I."""

    num_frames: int = 1
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    negative_prompt: str = ""
