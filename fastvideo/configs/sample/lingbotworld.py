# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from fastvideo.configs.sample.wan import Wan2_2_I2V_A14B_SamplingParam


@dataclass
class LingBotWorld_SamplingParam(Wan2_2_I2V_A14B_SamplingParam):
    guidance_scale: float = 3.5  # high_noise
    guidance_scale_2: float = 3.5  # low_noise
    num_inference_steps: int = 40
    fps: int = 16
    # NOTE(will): default boundary timestep is tracked by PipelineConfig, but
    # can be overridden during sampling
