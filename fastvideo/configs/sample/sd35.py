# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class SD35SamplingParam(SamplingParam):

    prompt: str | None = "a photo of a cat"
    negative_prompt: str = ""

    num_videos_per_prompt: int = 1
    seed: int = 0

    num_frames: int = 1
    height: int = 512
    width: int = 512
    fps: int = 1

    num_inference_steps: int = 28
    guidance_scale: float = 6.0
