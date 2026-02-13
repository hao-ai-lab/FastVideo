# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class LucyEditSamplingParam(SamplingParam):
    """Sampling parameters for Lucy-Edit-Dev video editing model.

    Based on the reference implementation defaults from the HuggingFace
    model card at https://huggingface.co/decart-ai/Lucy-Edit-Dev.
    """

    # Video parameters
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 24

    # Denoising stage
    guidance_scale: float = 5.0
    negative_prompt: str = ""
    num_inference_steps: int = 50
