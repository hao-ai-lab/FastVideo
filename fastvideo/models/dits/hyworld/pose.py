# Some functions from HY-WorldPlay/hyvideo/generate.py

"""
Pose processing utilities for HYWorld video generation.

This module provides functions to convert camera poses to model input tensors,
including viewmats, intrinsics, and action labels.

Adapted from HY-WorldPlay: https://github.com/Tencent-Hunyuan/HY-WorldPlay
"""

import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .trajectory import generate_camera_trajectory_local


# Mapping from one-hot action encoding to single label
mapping = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}

# Default camera intrinsic matrix (for 1920x1080 resolution)
DEFAULT_INTRINSIC = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
]

# Default movement speeds
DEFAULT_FORWARD_SPEED = 0.08  # units per frame
DEFAULT_YAW_SPEED = np.deg2rad(3)  # radians per frame
DEFAULT_PITCH_SPEED = np.deg2rad(3)  # radians per frame


def one_hot_to_one_dimension(one_hot: torch.Tensor) -> torch.Tensor:
    """Convert one-hot action encoding to single dimension labels."""
    return torch.tensor([mapping[tuple(row.tolist())] for row in one_hot])


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

def pose_string_to_json(
    pose_string: str,
    intrinsic: list[list[float]] | None = None,
) -> dict:
    """
    Convert pose string to pose JSON format.
    
    Args:
        pose_string: Comma-separated pose commands
        intrinsic: Camera intrinsic matrix (default: DEFAULT_INTRINSIC from trajectory)
        
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

def pose_to_input(
    pose_data: str | dict,
    latent_num: int,
    tps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert pose data to model input tensors.
    
    Args:
        pose_data: One of:
            - str ending with '.json': path to JSON file
            - str: pose string (e.g., "w-3, right-0.5, d-4")
            - dict: pose JSON data
        latent_num: Number of latents (frames in latent space)
        tps: Third-person mode flag
        
    Returns:
        Tuple of (viewmats, intrinsics, action_labels):
            - viewmats: World-to-camera matrices [T, 4, 4]
            - intrinsics: Normalized camera intrinsics [T, 3, 3]
            - action_labels: Action labels for each frame [T]
    """
    # Handle different input types
    if isinstance(pose_data, str):
        if pose_data.endswith(".json"):
            # Load from JSON file
            with open(pose_data) as f:
                pose_json = json.load(f)
        else:
            # Parse pose string
            pose_json = pose_string_to_json(pose_data)
    elif isinstance(pose_data, dict):
        pose_json = pose_data
    else:
        raise ValueError(
            f"Invalid pose_data type: {type(pose_data)}. Expected str or dict."
        )

    pose_keys = list(pose_json.keys())
    latent_num_from_pose = len(pose_keys)
    assert latent_num_from_pose == latent_num, (
        f"pose corresponds to {latent_num_from_pose * 4 - 3} frames, num_frames "
        f"must be set to {latent_num_from_pose * 4 - 3} to ensure alignment."
    )

    intrinsic_list = []
    w2c_list = []
    for i in range(latent_num):
        t_key = pose_keys[i]
        c2w = np.array(pose_json[t_key]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)
        
        # Normalize intrinsics
        intrinsic = np.array(pose_json[t_key]["K"])
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        intrinsic_list.append(intrinsic)

    w2c_list = np.array(w2c_list)
    intrinsic_list = torch.tensor(np.array(intrinsic_list))

    # Compute relative camera-to-world transforms
    c2ws = np.linalg.inv(w2c_list)
    C_inv = np.linalg.inv(c2ws[:-1])
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0, ...] = c2ws[0, ...]
    relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
    
    # Initialize one-hot action encodings
    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

    move_norm_valid = 0.0001
    for i in range(1, relative_c2w.shape[0]):
        move_dirs = relative_c2w[i, :3, 3]  # direction vector
        move_norms = np.linalg.norm(move_dirs)
        
        if move_norms > move_norm_valid:  # threshold for movement
            move_norm_dirs = move_dirs / move_norms
            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
            trans_angles_deg = angles_rad * (180.0 / np.pi)  # convert to degrees
        else:
            trans_angles_deg = np.zeros(3)

        R_rel = relative_c2w[i, :3, :3]
        r = R.from_matrix(R_rel)
        rot_angles_deg = r.as_euler("xyz", degrees=True)

        # Determine movement and rotation actions
        if move_norms > move_norm_valid:  # threshold for movement
            if (not tps) or (
                tps and abs(rot_angles_deg[1]) < 5e-2 and abs(rot_angles_deg[0]) < 5e-2
            ):
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # forward
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # backward

                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # right
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # left

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1  # right
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1  # left

        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1  # up
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1  # down

    trans_one_hot = torch.tensor(trans_one_hot)
    rotate_one_hot = torch.tensor(rotate_one_hot)

    # Convert one-hot to single-dimension labels
    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
    action_one_label = trans_one_label * 9 + rotate_one_label

    return (
        torch.as_tensor(w2c_list),
        torch.as_tensor(intrinsic_list),
        action_one_label,
    )


def camera_center_normalization(w2c: np.ndarray) -> np.ndarray:
    """Normalize camera centers relative to the first camera."""
    c2w = np.linalg.inv(w2c)
    C0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    return np.linalg.inv(c2w_aligned)



def parse_pose_string_to_actions(pose_string: str, fps: int = 24) -> list[dict]:
    """
    Parse pose string to frame-level action timeline.
    
    Format: pose string uses latent counts, where:
    - 1 latent = 4 frames
    - Special rule: first frame of entire video is extra (frame 0)
    - Example: "w-4,d-4" means:
      - w-4: forward for frames 0-16 (17 frames total: 1 extra + 4*4)
      - d-4: right for frames 17-32 (16 frames total: 4*4)
    
    Args:
        pose_string: Comma-separated pose commands (e.g., "w-4,d-4")
        fps: Frames per second for video (default: 24)
        
    Returns:
        List of dicts with action values for each frame
    """
    commands = [cmd.strip() for cmd in pose_string.split(",")]

    frame_actions = []
    is_first_command = True

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
            num_latents = int(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration in command: {cmd}")

        # Convert latents to frames
        # First command gets 1 extra frame (the special frame 0)
        if is_first_command:
            num_frames = 1 + num_latents * 4
            is_first_command = False
        else:
            num_frames = num_latents * 4

        # Map action to action values
        action_values = {"forward": 0, "left": 0, "yaw": 0, "pitch": 0}

        if action == "w":
            action_values["forward"] = 1
        elif action == "s":
            action_values["forward"] = -1
        elif action == "a":
            action_values["left"] = 1
        elif action == "d":
            action_values["left"] = -1
        elif action == "up":
            action_values["pitch"] = 1
        elif action == "down":
            action_values["pitch"] = -1
        elif action == "left":
            action_values["yaw"] = -1
        elif action == "right":
            action_values["yaw"] = 1
        else:
            raise ValueError(f"Unknown action: {action}")

        # Add frame-level actions
        for _ in range(num_frames):
            frame_actions.append(action_values.copy())

    return frame_actions


def compute_latent_num(num_frames: int) -> int:
    """
    Compute the number of latents from number of frames.
    
    Formula: num_frames = (latent_num - 1) * 4 + 1
    So: latent_num = (num_frames - 1) // 4 + 1
    
    Args:
        num_frames: Number of video frames
        
    Returns:
        Number of latents
    """
    return (num_frames - 1) // 4 + 1


def compute_num_frames(latent_num: int) -> int:
    """
    Compute the number of frames from number of latents.
    
    Formula: num_frames = (latent_num - 1) * 4 + 1
    
    Args:
        latent_num: Number of latents
        
    Returns:
        Number of video frames
    """
    return (latent_num - 1) * 4 + 1
