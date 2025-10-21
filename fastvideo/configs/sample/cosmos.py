# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class Cosmos_Predict2_2B_Video2World_SamplingParam(SamplingParam):
    # Video parameters
    height: int = 704
    width: int = 1280
    num_frames: int = 93
    fps: int = 16

    # Denoising stage
    guidance_scale: float = 7.0
    negative_prompt: str = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
    num_inference_steps: int = 35
