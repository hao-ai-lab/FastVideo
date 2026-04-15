# SPDX-License-Identifier: Apache-2.0
"""Sampling profiles for Cosmos and Cosmos 2.5 models."""
from fastvideo.configs.sample.profiles import ModelProfile

_COSMOS_NEGATIVE_PROMPT = ("The video captures a series of frames showing ugly scenes, "
                           "static with no motion, motion blur, over-saturation, "
                           "shaky footage, low resolution, grainy texture, "
                           "pixelated images, poorly lit areas, underexposed and "
                           "overexposed scenes, poor color balance, washed out colors, "
                           "choppy sequences, jerky movements, low frame rate, "
                           "artifacting, color banding, unnatural transitions, "
                           "outdated special effects, fake elements, unconvincing "
                           "visuals, poorly edited content, jump cuts, visual noise, "
                           "and flickering. Overall, the video is of poor quality.")

COSMOS_PREDICT2_2B = ModelProfile(
    name="cosmos_predict2_2b",
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 93,
        "fps": 16,
        "guidance_scale": 7.0,
        "num_inference_steps": 35,
        "negative_prompt": _COSMOS_NEGATIVE_PROMPT,
    },
)

COSMOS25_PREDICT2_2B = ModelProfile(
    name="cosmos25_predict2_2b",
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 77,
        "fps": 24,
        "seed": 0,
        "guidance_scale": 7.0,
        "num_inference_steps": 35,
        "negative_prompt": _COSMOS_NEGATIVE_PROMPT,
    },
)
