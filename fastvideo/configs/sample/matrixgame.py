# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class MatrixGame2SamplingParam(SamplingParam):
    height: int = 352
    width: int = 640
    num_frames: int = 57
    fps: int = 25
    guidance_scale: float = 1.0
    num_inference_steps: int = 3
    negative_prompt: str | None = None


@dataclass
class MatrixGame3SamplingParam(SamplingParam):
    height: int = 720
    width: int = 1280
    num_frames: int = 57
    fps: int = 25
    guidance_scale: float = 1.0
    num_inference_steps: int = 3
    negative_prompt: str | None = None
    num_iterations: int = 1
    use_base_model: bool = False
