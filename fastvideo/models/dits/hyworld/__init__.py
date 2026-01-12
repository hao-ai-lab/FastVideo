# SPDX-License-Identifier: Apache-2.0
"""
HyWorld (HY-WorldPlay) model components for FastVideo.

This module provides:
- HyWorldTransformer3DModel: The main transformer model with ProPE and action conditioning
- Camera trajectory generation utilities
- Pose processing utilities for model input
"""

from .hyworld import HyWorldTransformer3DModel, HyWorldDoubleStreamBlock

# Trajectory generation utilities
from .trajectory import (
    generate_camera_trajectory_local,
    parse_pose_string,
    pose_string_to_json,
    save_trajectory_json,
    rot_x,
    rot_y,
    rot_z,
    DEFAULT_FORWARD_SPEED,
    DEFAULT_YAW_SPEED,
    DEFAULT_PITCH_SPEED,
    DEFAULT_INTRINSIC,
)

# Pose processing utilities
from .pose import (
    pose_to_input,
    parse_pose_string_to_actions,
    compute_latent_num,
    compute_num_frames,
    one_hot_to_one_dimension,
    camera_center_normalization,
    ACTION_MAPPING,
)

__all__ = [
    # Model
    "HyWorldTransformer3DModel",
    "HyWorldDoubleStreamBlock",
    # Trajectory
    "generate_camera_trajectory_local",
    "parse_pose_string",
    "pose_string_to_json",
    "save_trajectory_json",
    "rot_x",
    "rot_y",
    "rot_z",
    "DEFAULT_FORWARD_SPEED",
    "DEFAULT_YAW_SPEED",
    "DEFAULT_PITCH_SPEED",
    "DEFAULT_INTRINSIC",
    # Pose
    "pose_to_input",
    "parse_pose_string_to_actions",
    "compute_latent_num",
    "compute_num_frames",
    "one_hot_to_one_dimension",
    "camera_center_normalization",
    "ACTION_MAPPING",
]
