# SPDX-License-Identifier: Apache-2.0
"""Sampling profiles for GEN3C models."""
from fastvideo.configs.sample.profiles import ModelProfile

GEN3C_COSMOS_7B = ModelProfile(
    name="gen3c_cosmos_7b",
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 35,
        "trajectory_type": "left",
        "movement_distance": 0.3,
        "camera_rotation": "center_facing",
    },
)
