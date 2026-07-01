# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

_ACTION_TO_MOTION = {
    "w": "forward",
    "a": "left",
    "d": "right",
    "s": "backward",
    "j": "left_rot",
    "l": "right_rot",
    "i": "up_rot",
    "k": "down_rot",
}
_TRANSLATION_BASE_UNIT = 1.0
_ROTATION_BASE_UNIT = 10.0
_INTRINSIC_ROW = [0.8, 0.5, 0.5, 0.5]


@dataclass
class DreamXCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    w2c_mat: np.ndarray

    @property
    def c2w_mat(self) -> np.ndarray:
        return np.linalg.inv(self.w2c_mat)

    @classmethod
    def from_pose_row(cls, row: list[float]) -> DreamXCamera:
        w2c_mat = np.eye(4, dtype=np.float64)
        w2c_mat[:3, :] = np.asarray(row[7:], dtype=np.float64).reshape(3, 4)
        return cls(
            fx=float(row[1]),
            fy=float(row[2]),
            cx=float(row[3]),
            cy=float(row[4]),
            w2c_mat=w2c_mat,
        )


def _translation_step(motion_type: str, current_pose: dict[str, np.ndarray], value: float, duration: int) -> np.ndarray:
    if motion_type in ("forward", "backward"):
        yaw = np.radians(current_pose["rotation"][1])
        pitch = np.radians(current_pose["rotation"][0])
        forward = np.array([-math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
        direction = 1 if motion_type == "forward" else -1
        return forward * value * direction / duration
    if motion_type in ("left", "right"):
        yaw = np.radians(current_pose["rotation"][1])
        right = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
        direction = -1 if motion_type == "left" else 1
        return right * value * direction / duration
    return np.zeros(3)


def _rotation_step(motion_type: str, value: float, duration: int) -> np.ndarray:
    if not motion_type.endswith("rot"):
        return np.zeros(3)
    axis = motion_type.split("_")[0]
    rotation = np.zeros(3)
    if axis == "left":
        rotation[1] = value
    elif axis == "right":
        rotation[1] = -value
    elif axis == "up":
        rotation[0] = -value
    elif axis == "down":
        rotation[0] = value
    return rotation / duration


def _euler_to_quaternion(angles: np.ndarray) -> list[float]:
    pitch, yaw, roll = np.radians(angles)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return [
        cy * cp * cr + sy * sp * sr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
        cy * cp * sr - sy * sp * cr,
    ]


def _quaternion_to_rotation_matrix(quaternion: list[float]) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
    ])


def _pose_rows_from_actions(action_seq: list[str], action_speed_list: list[float], duration: int) -> list[list[float]]:
    if len(action_seq) != len(action_speed_list):
        raise ValueError("action_seq and action_speed_list must have the same length")

    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    current_pose = {
        "position": np.array([0.0, 0.0, 0.0]),
        "rotation": np.array([0.0, 0.0, 0.0]),
    }

    for action_id, speed in zip(action_seq, action_speed_list, strict=True):
        motion_types = [_ACTION_TO_MOTION[key] for key in list(action_id)]
        translation_step = np.zeros(3)
        rotation_step = np.zeros(3)
        for motion_type in motion_types:
            translation_step += _translation_step(
                motion_type, current_pose, float(speed) * _TRANSLATION_BASE_UNIT, duration
            )
            rotation_step += _rotation_step(motion_type, float(speed) * _ROTATION_BASE_UNIT, duration)

        segment_positions = []
        segment_rotations = []
        for index in range(1, duration + 1):
            segment_positions.append(current_pose["position"] + translation_step * index)
            segment_rotations.append(current_pose["rotation"] + rotation_step * index)
        current_pose["position"] = segment_positions[-1].copy()
        current_pose["rotation"] = segment_rotations[-1].copy()
        positions.extend(segment_positions)
        rotations.extend(segment_rotations)

    rows: list[list[float]] = [[0.0] + _INTRINSIC_ROW + [0.0, 0.0] + [1.0, 0.0, 0.0, 0.0,
                                                                       0.0, 1.0, 0.0, 0.0,
                                                                       0.0, 0.0, 1.0, 0.0]]
    for index, (position, rotation) in enumerate(zip(positions, rotations, strict=False)):
        rotation_matrix = _quaternion_to_rotation_matrix(_euler_to_quaternion(rotation))
        translation = -rotation_matrix @ position
        extrinsic = np.hstack([rotation_matrix, translation.reshape(3, 1)])
        rows.append([float(index)] + _INTRINSIC_ROW + [0.0, 0.0] + extrinsic.flatten().tolist())
    return rows


def _interpolate_camera_poses(
    cameras: list[DreamXCamera],
    src_indices: np.ndarray,
    tgt_indices: np.ndarray,
) -> list[DreamXCamera]:
    src_rot_mat = np.array([camera.w2c_mat[:3, :3] for camera in cameras])
    src_trans_vec = np.array([camera.w2c_mat[:3, 3] for camera in cameras])

    dets = np.linalg.det(src_rot_mat)
    flip_handedness = dets.size > 0 and np.median(dets) < 0.0
    if flip_handedness:
        flip_mat = np.diag([1.0, 1.0, -1.0]).astype(src_rot_mat.dtype)
        src_rot_mat = src_rot_mat @ flip_mat

    trans = interp1d(src_indices, src_trans_vec, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")(
        tgt_indices
    )
    quats = Rotation.from_matrix(src_rot_mat).as_quat().copy()
    for index in range(1, len(quats)):
        if np.dot(quats[index], quats[index - 1]) < 0:
            quats[index] = -quats[index]
    rot = Slerp(src_indices, Rotation.from_quat(quats))(tgt_indices).as_matrix()
    if flip_handedness:
        rot = rot @ flip_mat

    ref = cameras[0]
    result = []
    for index in range(len(tgt_indices)):
        w2c_mat = np.eye(4, dtype=np.float64)
        w2c_mat[:3, :] = np.hstack([rot[index], trans[index].reshape(3, 1)])
        result.append(DreamXCamera(ref.fx, ref.fy, ref.cx, ref.cy, w2c_mat))
    return result


def _relative_c2w_poses(cameras: list[DreamXCamera]) -> np.ndarray:
    abs_w2cs = [camera.w2c_mat for camera in cameras]
    abs_c2ws = [camera.c2w_mat for camera in cameras]
    target_cam_c2w = np.eye(4, dtype=np.float64)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    poses = [target_cam_c2w] + [abs2rel @ c2w for c2w in abs_c2ws[1:]]
    return np.asarray(poses, dtype=np.float32)


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    output = torch.zeros_like(transforms)
    output[..., :3, :3] = rotation_inv
    output[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_inv, transforms[..., :3, 3])
    output[..., 3, 3] = 1.0
    return output


def build_dreamx_camera_condition(
    action_seq: list[str],
    action_speed_list: list[float],
    *,
    num_frames: int,
    height: int,
    width: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    del height, width  # DreamX-World-5B-Cam uses fixed normalized intrinsics.
    duration = math.ceil(num_frames / len(action_seq))
    rows = _pose_rows_from_actions(action_seq, action_speed_list, duration)[:num_frames]
    cameras = [DreamXCamera.from_pose_row(row) for row in rows]

    latent_frame_count = 1 + (len(cameras) - 1) // 4
    src_indices = np.arange(len(cameras), dtype=np.float64)
    tgt_indices = np.linspace(0, len(cameras) - 1, latent_frame_count)
    cameras = _interpolate_camera_poses(cameras, src_indices, tgt_indices)

    c2ws = torch.as_tensor(_relative_c2w_poses(cameras), dtype=dtype, device=device)
    viewmats = _invert_se3(c2ws)

    intrinsics = torch.zeros((latent_frame_count, 3, 3), dtype=dtype, device=device)
    intrinsics[:, 0, 0] = 969.6969696969696 / (960.0 * 2)
    intrinsics[:, 1, 1] = 969.6969696969696 / (540.0 * 2)
    intrinsics[:, 2, 2] = 1.0
    return {"viewmats": viewmats, "K": intrinsics}
