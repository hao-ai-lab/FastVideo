# SPDX-License-Identifier: Apache-2.0
"""
Sampling parameters for Hunyuan-GameCraft.

HunyuanGameCraft is a camera-controllable video generation model
built on HunyuanVideo with CameraNet for pose conditioning.
"""
from dataclasses import dataclass, field
from typing import Any

import torch

from fastvideo.configs.sample.base import SamplingParam
from fastvideo.configs.sample.teacache import TeaCacheParams


@dataclass
class HunyuanGameCraftSamplingParam(SamplingParam):
    """
    Default sampling parameters for Hunyuan-GameCraft.
    
    The model generates videos at 704x1280 resolution with 34 frames (~1.4s at 24fps).
    Camera poses are provided as 6D Plücker coordinates for controllable camera motion.
    
    NOTE: The official implementation uses target_length=34 for single-image I2V mode,
    which produces 10 latent frames with their VAE formula (34-2)//4+2=10.
    FastVideo's VAE uses a different formula: (n-1)//4+1. To get the same
    10 latent frames, we need num_frames=37: (37-1)//4+1=10.
    
    Getting exactly 10 latent frames is critical because:
    - The transformer has specific camera conditioning for latent_len=10
    - At latent_len=9 it takes a completely different (wrong) code path
    - RoPE positional encodings depend on the temporal dimension
    """
    num_inference_steps: int = 50
    
    # Video dimensions (GameCraft default resolution)
    # Must be 37 to get 10 latent frames: (37-1)//4+1 = 10
    # (official uses 34 with their formula (34-2)//4+2 = 10)
    num_frames: int = 37
    height: int = 704
    width: int = 1280
    fps: int = 24
    
    # Guidance scale for classifier-free guidance
    guidance_scale: float = 6.0
    
    # Camera states for CameraNet conditioning
    # Shape: [num_frames, 6, height, width] - Plücker coordinates
    camera_states: torch.Tensor | None = None
    
    # TeaCache parameters for faster inference
    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.15,
            coefficients=[
                7.33226126e+02, -4.01131952e+02, 6.75869174e+01,
                -3.14987800e+00, 9.61237896e-02
            ]
        )
    )


@dataclass
class HunyuanGameCraftI2VSamplingParam(HunyuanGameCraftSamplingParam):
    """
    Sampling parameters for Hunyuan-GameCraft Image-to-Video.
    
    Same as T2V but with image conditioning for the first frame.
    """
    pass


@dataclass
class FastHunyuanGameCraftSamplingParam(HunyuanGameCraftSamplingParam):
    """
    Fast sampling parameters for distilled Hunyuan-GameCraft models.
    
    Uses fewer inference steps for faster generation.
    """
    num_inference_steps: int = 6
