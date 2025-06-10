# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.v1.configs.sample.base import SamplingParam
from fastvideo.v1.configs.sample.teacache import TeaCacheParams


@dataclass
class HunyuanSamplingParam(SamplingParam):
    num_inference_steps: int = 50

    num_frames: int = 125
    height: int = 720
    width: int = 1280
    fps: int = 24

    guidance_scale: float = 1.0

    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.15,
            coefficients=[
                7.33226126e+02, -4.01131952e+02, 6.75869174e+01,
                -3.14987800e+00, 9.61237896e-02
            ]))


@dataclass
class FastHunyuanSamplingParam(HunyuanSamplingParam):
    num_inference_steps: int = 6
