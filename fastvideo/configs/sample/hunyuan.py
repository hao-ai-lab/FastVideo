# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class HunyuanSamplingParam(SamplingParam):
    num_inference_steps: int = 50

    num_frames: int = 125
    height: int = 720
    width: int = 1280
    fps: int = 24

    guidance_scale: float = 1.0


@dataclass
class FastHunyuanSamplingParam(HunyuanSamplingParam):
    num_inference_steps: int = 6
