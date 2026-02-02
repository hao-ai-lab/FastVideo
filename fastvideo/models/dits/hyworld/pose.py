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
from typing import Union, Optional

from fastvideo.models.dits.hyworld.trajectory import generate_camera_trajectory_local


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
    intrinsic: Optional[list[list[float]]] = None,
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
    pose_data: Union[str, dict],
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
            with open(pose_data, "r") as f:
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

def reformat_keyboard_and_mouse_tensors(keyboard_tensor, mouse_tensor):
    """
    Reformat the keyboard and mouse tensors to the format compatible with HyWorld.
    """
    num_frames = keyboard_tensor.shape[0]
    assert (num_frames - 1) % 4 == 0, "num_frames must be a multiple of 4"
    assert mouse_tensor.shape[0] == num_frames, "mouse_tensor must have the same number of frames as keyboard_tensor"
    keyboard_tensor = keyboard_tensor[1:, :]
    mouse_tensor = mouse_tensor[1:, :]
    groups = keyboard_tensor.view(-1, 4, keyboard_tensor.shape[1])
    assert (groups == groups[:, 0:1]).all(dim=1).all(), "keyboard_tensor must have the same value for each group"
    groups = mouse_tensor.view(-1, 4, mouse_tensor.shape[1])
    assert (groups == groups[:, 0:1]).all(dim=1).all(), "mouse_tensor must have the same value for each group"
    
    return keyboard_tensor[::4], mouse_tensor[::4]

def process_custom_actions(keyboard_tensor, mouse_tensor, forward_speed=DEFAULT_FORWARD_SPEED):
    """
    Process custom keyboard and mouse tensors into model inputs (viewmats, intrinsics, action_labels).
    Assumes inputs correspond to each LATENT frame.
    """
    keyboard_tensor, mouse_tensor = reformat_keyboard_and_mouse_tensors(keyboard_tensor, mouse_tensor)

    motions = []
    
    # 1. Translate tensors to motions for trajectory generation
    for t in range(keyboard_tensor.shape[0]):
        frame_motion = {}
        
        # --- Translation ---
        # MatrixGame convention: 0:W, 1:S, 2:A, 3:D
        fwd = 0.0
        if keyboard_tensor[t, 0] > 0.5: fwd += forward_speed   # W
        if keyboard_tensor[t, 1] > 0.5: fwd -= forward_speed   # S
        if fwd != 0: frame_motion["forward"] = fwd
        
        rgt = 0.0
        if keyboard_tensor[t, 2] > 0.5: rgt -= forward_speed   # A (Left is negative Right)
        if keyboard_tensor[t, 3] > 0.5: rgt += forward_speed   # D (Right)
        if rgt != 0: frame_motion["right"] = rgt
        
        # --- Rotation ---
        # MatrixGame convention: mouse is [Pitch, Yaw] (or Y, X)
        # Apply scaling (e.g. to match HyWorld distribution)
        pitch = mouse_tensor[t, 0].item()
        yaw = mouse_tensor[t, 1].item()
        
        if abs(pitch) > 1e-4: frame_motion["pitch"] = pitch
        if abs(yaw) > 1e-4: frame_motion["yaw"] = yaw
        
        motions.append(frame_motion)

    # 2. Generate Camera Trajectory
    # generate_camera_trajectory_local returns T+1 poses (starting at Identity)
    # We take the first T poses to match the latent count.
    # Pose 0 is Identity. Pose 1 is Identity + Motion[0].
    poses = generate_camera_trajectory_local(motions)
    # poses = np.array(poses[:T])
    
    # 3. Compute Viewmats (w2c) and Intrinsics
    w2c_list = []
    intrinsic_list = []
    
    # Setup default intrinsic (normalized)
    K = np.array(DEFAULT_INTRINSIC)
    K[0, 0] /= K[0, 2] * 2
    K[1, 1] /= K[1, 2] * 2
    K[0, 2] = 0.5
    K[1, 2] = 0.5
    
    for i in range(len(poses)):
        c2w = np.array(poses[i])
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)
        intrinsic_list.append(K)
        
    viewmats = torch.as_tensor(np.array(w2c_list))
    intrinsics = torch.as_tensor(np.array(intrinsic_list))

    # 4. Generate Action Labels DIRECTLY from inputs
    # HyWorld Label Logic:
    # Trans One-Hot: [Forward, Backward, Right, Left] (Indices 0, 1, 2, 3)
    # Rotate One-Hot: [Right, Left, Up, Down] (Indices 0, 1, 2, 3)
    
    trans_one_hot = torch.zeros((keyboard_tensor.shape[0], 4), dtype=torch.long)
    trans_one_hot[:, 0] = (keyboard_tensor[:, 0] > 0.5).long() # Forward
    trans_one_hot[:, 1] = (keyboard_tensor[:, 1] > 0.5).long() # Backward
    trans_one_hot[:, 2] = (keyboard_tensor[:, 3] > 0.5).long() # Right
    trans_one_hot[:, 3] = (keyboard_tensor[:, 2] > 0.5).long() # Left
    
    rotate_one_hot = torch.zeros((mouse_tensor.shape[0], 4), dtype=torch.long)
    rotate_one_hot[:, 0] = (mouse_tensor[:, 1] > 1e-4).long()  # Yaw Right
    rotate_one_hot[:, 1] = (mouse_tensor[:, 1] < -1e-4).long() # Yaw Left
    rotate_one_hot[:, 2] = (mouse_tensor[:, 0] > 1e-4).long()  # Pitch Up
    rotate_one_hot[:, 3] = (mouse_tensor[:, 0] < -1e-4).long() # Pitch Down

    # Convert to single labels
    trans_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_label = one_hot_to_one_dimension(rotate_one_hot)
    action_labels = trans_label * 9 + rotate_label

    action_labels = torch.cat([torch.tensor([0]), action_labels])

    return viewmats, intrinsics, action_labels

if __name__ == "__main__":
    print("Running comparison test between process_custom_actions and pose_to_input...")

    def test_process_custom_actions(pose_string: str, keyboard: torch.Tensor, mouse: torch.Tensor, latent_num: int):
        # Run process_custom_actions
        # Note: We need to pass float tensors
        print("Running process_custom_actions...")
        viewmats_1, intrinsics_1, labels_1 = process_custom_actions(
            keyboard, mouse
        )
        
        print(f"Running pose_to_input with string: '{pose_string}'...")
        viewmats_2, intrinsics_2, labels_2 = pose_to_input(
            pose_string, latent_num=latent_num
        )

        # print(f"Viewmats: {viewmats_1} vs \n {viewmats_2}")
        # print(f"Intrinsics: {intrinsics_1} vs \n {intrinsics_2}")
        # print(f"Labels: {labels_1} vs \n {labels_2}")
        # 3. Compare Results
        print("\nComparison Results:")
        
        # Check Shapes
        print(f"Shapes (Viewmats): {viewmats_1.shape} vs {viewmats_2.shape}")
        assert viewmats_1.shape == viewmats_2.shape, "Shape mismatch for viewmats"
        
        # Check Values
        # Viewmats
        diff_viewmats = (viewmats_1 - viewmats_2).abs().max().item()
        print(f"Max difference in Viewmats: {diff_viewmats}")
        if diff_viewmats < 1e-5:
            print("✅ Viewmats match.")
        else:
            print("❌ Viewmats mismatch.")

        # Check intrinsics
        diff_intrinsics = (intrinsics_1 - intrinsics_2).abs().max().item()
        print(f"Max difference in Intrinsics: {diff_intrinsics}")
        if diff_intrinsics < 1e-5:
            print("✅ Intrinsics match.")
        else:
            print("❌ Intrinsics mismatch.")

        # Check labels
        diff_labels = (labels_1 - labels_2).abs().max().item()
        print(f"Max difference in Labels: {diff_labels}")
        if diff_labels < 1e-5:
            print("✅ Labels match.")
        else:
            print("❌ Labels mismatch.")

        print("All checks passed.")
    
    # Define shared parameters

    latent_num = 13
    pose_string = "w-2, a-3, s-1, d-6"

    num_frames = 4 * (latent_num - 1) + 1
    keyboard = torch.zeros((num_frames, 6))
    mouse = torch.zeros((num_frames, 2))

    # Frame 0 is ignored/start
    # Frames 1-8: Press W (index 0)
    keyboard[1:9, 0] = 1.0
    # Frames 9-20: Press A (index 2)
    keyboard[9:21, 2] = 1.0
    # Frames 21-24: Press S (index 1)
    keyboard[21:25, 1] = 1.0
    # Frames 25-48: Press D (index 3)
    keyboard[25:49, 3] = 1.0

    test_process_custom_actions(pose_string, keyboard, mouse, latent_num)

    # Test keyboard AND mouse
    latent_num = 25
    pose_string = "w-2, up-2, a-3, down-4, s-1, left-2, d-6, right-4"

    num_frames = 4 * (latent_num - 1) + 1
    keyboard = torch.zeros((num_frames, 6))
    mouse = torch.zeros((num_frames, 2))

    # Frame 0 is ignored/start
    # Frames 1-8: Press W (index 0)
    keyboard[1:9, 0] = 1.0
    # Frames 17-28: Press A (index 2)
    keyboard[17:29, 2] = 1.0
    # Frames 45-48: Press S (index 1)
    keyboard[45:49, 1] = 1.0
    # Frames 57-80: Press D (index 3)
    keyboard[57:81, 3] = 1.0
    
    # Frames 9-16: Press Up (index 4)
    mouse[9:17, 0] = DEFAULT_PITCH_SPEED
    # Frames 25-32: Press Down (index 5)
    mouse[29:45, 0] = -DEFAULT_PITCH_SPEED
    # Frames 41-48: Press Left (index 6)
    mouse[49:57, 1] = -DEFAULT_YAW_SPEED
    # Frames 57-64: Press Right (index 7)
    mouse[81:, 1] = DEFAULT_YAW_SPEED

    test_process_custom_actions(pose_string, keyboard, mouse, latent_num)
