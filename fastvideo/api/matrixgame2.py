# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.api.sampling_param import SamplingParam


@dataclass
class MatrixGame2SamplingParam(SamplingParam):
    height: int = 352
    width: int = 640
    num_frames: int = 57
    fps: int = 25
    guidance_scale: float = 1.0
    num_inference_steps: int = 3
    negative_prompt: str | None = None
