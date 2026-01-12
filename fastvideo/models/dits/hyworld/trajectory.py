# SPDX-License-Identifier: Apache-2.0
"""
Camera trajectory generation utilities for HyWorld video generation.

This module provides functions to generate camera trajectories from motion commands,
supporting various camera movements like forward/backward, yaw, pitch, and third-person rotation.

Adapted from HY-WorldPlay: https://github.com/Tencent-Hunyuan/HY-WorldPlay
"""

import json
import numpy as np
from typing import Optional


def rot_x(theta: float) -> np.ndarray:
    """Create rotation matrix for rotation around X-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta: float) -> np.ndarray:
    """Create rotation matrix for rotation around Y-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta: float) -> np.ndarray:
    """Create rotation matrix for rotation around Z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def generate_camera_trajectory_local(motions: list[dict]) -> list[np.ndarray]:
    """
    Generate camera trajectory from motion commands.
    
    Each motion is a dict that can contain:
    - "forward": Translation forward/backward (positive = forward)
    - "right": Translation left/right (positive = right)
    - "yaw": Rotation left/right (radians)
    - "pitch": Rotation up/down (radians)
    - "third_yaw": Third-person perspective rotation (radians)
    
    Args:
        motions: List of motion dictionaries
        
    Returns:
        List of 4x4 camera-to-world transformation matrices
    """
    poses = []
    T = np.eye(4)
    poses.append(T.copy())

    for move in motions:
        # Rotate (Left or Right)
        if "yaw" in move:
            R = rot_y(move["yaw"])
            T[:3, :3] = T[:3, :3] @ R

        # Rotate (Up or Down)
        if "pitch" in move:
            R = rot_x(move["pitch"])
            T[:3, :3] = T[:3, :3] @ R

        # Translation (Z-direction of the camera's local coordinate system)
        forward = move.get("forward", 0.0)
        if forward != 0:
            local_t = np.array([0, 0, forward])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Translation (X-direction of the camera's local coordinate system)
        right = move.get("right", 0.0)
        if right != 0:
            local_t = np.array([right, 0, 0])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Third Perspective Rotate (Left or Right)
        third_yaw = move.get("third_yaw", 0.0)
        if third_yaw != 0:
            theta = -third_yaw
            C = np.array([[1, 0.0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1.0], [0, 0, 0, 1]])
            c_origin = C.copy()
            # Rotation around the Y-axis
            R_y = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
            # Translation
            C[:3, :3] = C[:3, :3] @ R_y
            C[:3, 3] = R_y @ C[:3, 3]
            c_inv = np.linalg.inv(c_origin)
            c_relative = c_inv @ C
            T = T @ c_relative

        poses.append(T.copy())

    return poses


# Default movement speeds
DEFAULT_FORWARD_SPEED = 0.08  # units per frame
DEFAULT_YAW_SPEED = np.deg2rad(3)  # radians per frame
DEFAULT_PITCH_SPEED = np.deg2rad(3)  # radians per frame


def parse_pose_string(
    pose_string: str,
    forward_speed: float = DEFAULT_FORWARD_SPEED,
    yaw_speed: float = DEFAULT_YAW_SPEED,
    pitch_speed: float = DEFAULT_PITCH_SPEED,
) -> list[dict]:
    """
    Parse pose string to motions list.
    
    Format: "w-3, right-0.5, d-4"
    - w: forward movement
    - s: backward movement
    - a: left movement
    - d: right movement
    - up: pitch up rotation
    - down: pitch down rotation
    - left: yaw left rotation
    - right: yaw right rotation
    - number after dash: duration in frames/latents
    
    Args:
        pose_string: Comma-separated pose commands
        forward_speed: Movement amount per frame
        yaw_speed: Yaw rotation amount per frame (radians)
        pitch_speed: Pitch rotation amount per frame (radians)
        
    Returns:
        List of motion dictionaries for generate_camera_trajectory_local
    """
    motions = []
    commands = [cmd.strip() for cmd in pose_string.split(",")]

    for cmd in commands:
        if not cmd:
            continue

        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pose command: {cmd}. Expected format: 'action-duration'"
            )

        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration in command: {cmd}")

        num_frames = int(duration)

        # Parse action and create motion dicts
        if action == "w":
            # Forward
            for _ in range(num_frames):
                motions.append({"forward": forward_speed})
        elif action == "s":
            # Backward
            for _ in range(num_frames):
                motions.append({"forward": -forward_speed})
        elif action == "a":
            # Left
            for _ in range(num_frames):
                motions.append({"right": -forward_speed})
        elif action == "d":
            # Right
            for _ in range(num_frames):
                motions.append({"right": forward_speed})
        elif action == "up":
            # Pitch up
            for _ in range(num_frames):
                motions.append({"pitch": pitch_speed})
        elif action == "down":
            # Pitch down
            for _ in range(num_frames):
                motions.append({"pitch": -pitch_speed})
        elif action == "left":
            # Yaw left
            for _ in range(num_frames):
                motions.append({"yaw": -yaw_speed})
        elif action == "right":
            # Yaw right
            for _ in range(num_frames):
                motions.append({"yaw": yaw_speed})
        else:
            raise ValueError(
                f"Unknown action: {action}. "
                f"Supported actions: w, s, a, d, up, down, left, right"
            )

    return motions


# Default camera intrinsic matrix (for 1920x1080 resolution)
DEFAULT_INTRINSIC = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
]


def pose_string_to_json(
    pose_string: str,
    intrinsic: Optional[list[list[float]]] = None,
) -> dict:
    """
    Convert pose string to pose JSON format.
    
    Args:
        pose_string: Comma-separated pose commands
        intrinsic: Camera intrinsic matrix (default: DEFAULT_INTRINSIC)
        
    Returns:
        Dict with frame indices as keys, containing extrinsic and K (intrinsic) matrices
    """
    if intrinsic is None:
        intrinsic = DEFAULT_INTRINSIC
        
    motions = parse_pose_string(pose_string)
    poses = generate_camera_trajectory_local(motions)

    pose_json = {}
    for i, p in enumerate(poses):
        pose_json[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}

    return pose_json


def save_trajectory_json(
    pose_string: str,
    output_path: str,
    intrinsic: Optional[list[list[float]]] = None,
) -> None:
    """
    Generate and save camera trajectory to JSON file.
    
    Args:
        pose_string: Comma-separated pose commands
        output_path: Path to save the JSON file
        intrinsic: Camera intrinsic matrix (default: DEFAULT_INTRINSIC)
    """
    pose_json = pose_string_to_json(pose_string, intrinsic)
    with open(output_path, "w") as f:
        json.dump(pose_json, f, indent=4, ensure_ascii=False)
