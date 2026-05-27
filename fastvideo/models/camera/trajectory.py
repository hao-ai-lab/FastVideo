# SPDX-License-Identifier: Apache-2.0
"""
Camera trajectory generation for HunyuanGameCraft.

Generates Plücker coordinate embeddings from simple action commands
(forward, backward, left, right, rotations) for camera conditioning.

This is a self-contained implementation that does not depend on the
official Hunyuan-GameCraft-1.0 repository.
"""
import math

import numpy as np
import torch
from packaging import version as pver


# Action name -> motion type mapping
ACTION_DICT = {
    "w": "forward",
    "a": "left",
    "d": "right",
    "s": "backward",
    "forward": "forward",
    "backward": "backward",
    "left": "left",
    "right": "right",
    "left_rot": "left_rot",
    "right_rot": "right_rot",
    "up_rot": "up_rot",
    "down_rot": "down_rot",
}


def _custom_meshgrid(*args):
    """Torch meshgrid with consistent indexing."""
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def _generate_motion_segment(
    current_pose: dict,
    motion_type: str,
    value: float,
    duration: int = 30,
) -> tuple[list, list, dict]:
    """
    Generate camera motion segment.

    Args:
        current_pose: Dict with 'position' (xyz) and 'rotation' (pitch, yaw, roll).
        motion_type: One of 'forward', 'backward', 'left', 'right',
                     'left_rot', 'right_rot', 'up_rot', 'down_rot'.
        value: Translation (meters) or rotation (degrees).
        duration: Number of frames.

    Returns:
        positions: List of position arrays.
        rotations: List of rotation arrays.
        current_pose: Updated pose dict.
    """
    positions = []
    rotations = []

    if motion_type in ["forward", "backward"]:
        yaw_rad = np.radians(current_pose["rotation"][1])
        pitch_rad = np.radians(current_pose["rotation"][0])

        forward_vec = np.array([
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad),
        ])

        direction = 1 if motion_type == "forward" else -1
        total_move = forward_vec * value * direction
        step = total_move / duration

        for i in range(1, duration + 1):
            new_pos = current_pose["position"] + step * i
            positions.append(new_pos.copy())
            rotations.append(current_pose["rotation"].copy())

        current_pose["position"] = positions[-1]

    elif motion_type in ["left", "right"]:
        yaw_rad = np.radians(current_pose["rotation"][1])
        right_vec = np.array([math.cos(yaw_rad), 0, -math.sin(yaw_rad)])

        direction = -1 if motion_type == "right" else 1
        total_move = right_vec * value * direction
        step = total_move / duration

        for i in range(1, duration + 1):
            new_pos = current_pose["position"] + step * i
            positions.append(new_pos.copy())
            rotations.append(current_pose["rotation"].copy())

        current_pose["position"] = positions[-1]

    elif motion_type.endswith("rot"):
        axis = motion_type.split("_")[0]
        total_rotation = np.zeros(3)

        if axis == "left":
            total_rotation[0] = value
        elif axis == "right":
            total_rotation[0] = -value
        elif axis == "up":
            total_rotation[2] = -value
        elif axis == "down":
            total_rotation[2] = value

        step = total_rotation / duration

        for i in range(1, duration + 1):
            positions.append(current_pose["position"].copy())
            new_rot = current_pose["rotation"] + step * i
            rotations.append(new_rot.copy())

        current_pose["rotation"] = rotations[-1]

    return positions, rotations, current_pose


def _euler_to_quaternion(angles: np.ndarray) -> list[float]:
    """Convert Euler angles (pitch, yaw, roll in degrees) to quaternion."""
    pitch, yaw, roll = np.radians(angles)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return [qw, qx, qy, qz]


def _quaternion_to_rotation_matrix(q: list[float]) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
    ])


def _action_to_pose_list(action_id: str, value: float = 0.2, duration: int = 33) -> list[str]:
    """
    Convert an action ID to a list of pose strings.

    Args:
        action_id: Action identifier (e.g., 'w', 'forward', 'left_rot').
        value: Motion magnitude (translation in meters, rotation in degrees).
        duration: Number of frames.

    Returns:
        List of pose strings in the official GameCraft format.
    """
    all_positions = []
    all_rotations = []
    current_pose = {
        "position": np.array([0.0, 0.0, 0.0]),
        "rotation": np.array([0.0, 0.0, 0.0]),
    }
    intrinsic = [0.50505, 0.8979, 0.5, 0.5]

    motion_type = ACTION_DICT.get(action_id, action_id)
    positions, rotations, current_pose = _generate_motion_segment(
        current_pose, motion_type, value, duration
    )
    all_positions.extend(positions)
    all_rotations.extend(rotations)

    pose_list = []

    # First frame: identity pose
    row = [0] + intrinsic + [0, 0] + [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    first_row = " ".join(map(str, row))
    pose_list.append(first_row)

    for i, (pos, rot) in enumerate(zip(all_positions, all_rotations)):
        quat = _euler_to_quaternion(rot)
        R = _quaternion_to_rotation_matrix(quat)
        extrinsic = np.hstack([R, pos.reshape(3, 1)])

        row = [i] + intrinsic + [0, 0] + extrinsic.flatten().tolist()
        pose_list.append(" ".join(map(str, row)))

    return pose_list


class _Camera:
    """Camera parameters from a pose string."""

    def __init__(self, entry: list[float]):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def _get_relative_pose(cam_params: list[_Camera]) -> np.ndarray:
    """Convert camera parameters to relative poses."""
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    for pose in ret_poses:
        pose[:3, -1:] *= 10
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def _get_c2w(w2cs: list[np.ndarray], transform_matrix: np.ndarray) -> np.ndarray:
    """Convert w2c matrices to c2w with relative transform."""
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    abs2rel = target_cam_c2w @ w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    for pose in ret_poses:
        pose[:3, -1:] *= 2
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)


def _ray_condition(
    K: torch.Tensor,
    c2w: torch.Tensor,
    H: int,
    W: int,
    device: str | torch.device = "cpu",
    flip_flag: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Plücker coordinates from camera intrinsics and extrinsics.

    Args:
        K: Intrinsics [B, V, 4] (fx, fy, cx, cy).
        c2w: Camera-to-world matrices [B, V, 4, 4].
        H: Image height.
        W: Image width.
        device: Torch device.
        flip_flag: Optional flip flags [V].

    Returns:
        Plücker coordinates [B, V, H, W, 6].
    """
    B, V = K.shape[:2]

    j, i = _custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = _custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype),
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)
    return plucker


def create_camera_trajectory(
    action: str,
    height: int,
    width: int,
    num_frames: int,
    action_speed: float = 0.2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Create Plücker coordinate embeddings from an action command.

    Args:
        action: One of 'forward', 'backward', 'left', 'right',
                'left_rot', 'right_rot', 'up_rot', 'down_rot'
                (or shorthand 'w', 'a', 's', 'd').
        height: Video height in pixels.
        width: Video width in pixels.
        num_frames: Number of video frames.
        action_speed: Speed of motion (default 0.2).
        device: Torch device for output tensor.
        dtype: Torch dtype for output tensor.

    Returns:
        camera_states: [1, num_frames, 6, height, width] Plücker embeddings.
    """
    # Generate pose list from action
    poses = _action_to_pose_list(action, value=action_speed, duration=num_frames)

    # Parse poses
    poses_parsed = [pose.split(" ") for pose in poses]
    start_idx = 0
    sample_id = [start_idx + i for i in range(num_frames)]
    poses_parsed = [poses_parsed[i] for i in sample_id]

    # Convert to w2c matrices
    w2cs = [np.asarray([float(p) for p in pose[7:]]).reshape(3, 4) for pose in poses_parsed]
    transform_matrix = np.asarray(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    ).reshape(4, 4)
    last_row = np.zeros((1, 4))
    last_row[0, -1] = 1.0
    w2cs = [np.concatenate((w2c, last_row), axis=0) for w2c in w2cs]
    c2ws = _get_c2w(w2cs, transform_matrix)

    # Parse camera parameters
    cam_params = [[float(x) for x in pose] for pose in poses_parsed]
    assert len(cam_params) == num_frames
    cam_params = [_Camera(cam_param) for cam_param in cam_params]

    # Compute scaled intrinsics
    monst3r_w = cam_params[0].cx * 2
    monst3r_h = cam_params[0].cy * 2
    ratio_w, ratio_h = width / monst3r_w, height / monst3r_h
    intrinsics = np.asarray(
        [
            [
                cam_param.fx * ratio_w,
                cam_param.fy * ratio_h,
                cam_param.cx * ratio_w,
                cam_param.cy * ratio_h,
            ]
            for cam_param in cam_params
        ],
        dtype=np.float32,
    )
    intrinsics = torch.as_tensor(intrinsics)[None]  # [1, n_frame, 4]

    # Get relative poses
    c2w_poses = _get_relative_pose(cam_params)
    c2w = torch.as_tensor(c2w_poses)[None]  # [1, n_frame, 4, 4]

    # Compute Plücker embeddings
    flip_flag = torch.zeros(num_frames, dtype=torch.bool, device="cpu")
    plucker_embedding = _ray_condition(intrinsics, c2w, height, width, device="cpu", flip_flag=flip_flag)
    # [1, n_frame, H, W, 6] -> [1, n_frame, 6, H, W]
    plucker_embedding = plucker_embedding[0].permute(0, 3, 1, 2).contiguous()

    # Add batch dim and convert to target dtype/device
    # Shape: [n_frame, 6, H, W] -> [1, n_frame, 6, H, W]
    camera_states = plucker_embedding.unsqueeze(0).to(device=device, dtype=dtype)

    return camera_states
