# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.api.sampling_param import SamplingParam


@dataclass
class WanTrackSamplingParam(SamplingParam):
    """Sampling defaults for causal WanTrack Self-Forcing I2V."""

    height: int = 480
    width: int = 832
    num_frames: int = 121
    fps: int = 16
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    negative_prompt: str | None = None
