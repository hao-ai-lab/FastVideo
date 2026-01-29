# SPDX-License-Identifier: Apache-2.0
# Flux2 sampling params (image gen: 1 frame).
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class Flux2SamplingParam(SamplingParam):
    """Default sampling params for Flux2 image generation."""
    num_frames: int = 1
    num_inference_steps: int = 28
    guidance_scale: float = 4.0
    height: int = 1024
    width: int = 1024


@dataclass
class Flux2KleinSamplingParam(SamplingParam):
    """Sampling params for Flux2 Klein (distilled, 4-step, no guidance)."""
    num_frames: int = 1
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    height: int = 1024
    width: int = 1024
