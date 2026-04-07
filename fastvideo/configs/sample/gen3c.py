# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class Gen3C_Cosmos_7B_SamplingParam(SamplingParam):
    """Defaults for GEN3C (Cosmos-7B) camera-controlled video generation."""

    # Video parameters (matching official GEN3C defaults)
    height: int = 704
    width: int = 1280
    num_frames: int = 121
    fps: int = 24

    # Denoising stage
    guidance_scale: float = 1.0
    num_inference_steps: int = 35

    # GEN3C camera control defaults
    trajectory_type: str = "left"
    movement_distance: float = 0.3
    camera_rotation: str = "center_facing"
