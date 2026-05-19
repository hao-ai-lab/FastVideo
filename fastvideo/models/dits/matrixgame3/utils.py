from __future__ import annotations

import random

import numpy as np
import torch

from fastvideo.models.dits.lingbotworld.cam_utils import (
    compute_relative_poses,
    get_plucker_embeddings,
    interpolate_camera_poses,
)


WSAD_OFFSET = 12.35
DIAGONAL_OFFSET = 8.73
MOUSE_PITCH_SENSITIVITY = 15.0
MOUSE_YAW_SENSITIVITY = 15.0
MOUSE_THRESHOLD = 0.02


def compute_next_pose_from_action(current_pose, keyboard_action, mouse_action):
    x, y, z, pitch, yaw = current_pose
    w, s, a, d = keyboard_action[:4]
    mouse_x, mouse_y = mouse_action[:2]

    delta_pitch = MOUSE_PITCH_SENSITIVITY * mouse_x if abs(mouse_x) >= MOUSE_THRESHOLD else 0.0
    delta_yaw = MOUSE_YAW_SENSITIVITY * mouse_y if abs(mouse_y) >= MOUSE_THRESHOLD else 0.0
    new_pitch = pitch + delta_pitch
    new_yaw = yaw + delta_yaw

    new_yaw = (new_yaw + 180) % 360 - 180

    local_forward = 0.0
    if w > 0.5 and s < 0.5:
        local_forward = WSAD_OFFSET
    elif s > 0.5 and w < 0.5:
        local_forward = -WSAD_OFFSET

    local_right = 0.0
    if d > 0.5 and a < 0.5:
        local_right = WSAD_OFFSET
    elif a > 0.5 and d < 0.5:
        local_right = -WSAD_OFFSET

    if abs(local_forward) > 0.1 and abs(local_right) > 0.1:
        local_forward = np.sign(local_forward) * DIAGONAL_OFFSET
        local_right = np.sign(local_right) * DIAGONAL_OFFSET

    avg_yaw = float((yaw + new_yaw) / 2.0)
    yaw_rad = float(np.deg2rad(avg_yaw))
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    delta_x = cos_yaw * local_forward - sin_yaw * local_right
    delta_y = sin_yaw * local_forward + cos_yaw * local_right
    return np.array([x + delta_x, y + delta_y, z, new_pitch, new_yaw], dtype=np.float32)


def compute_all_poses_from_actions(keyboard_conditions, mouse_conditions):
    poses = np.zeros((len(keyboard_conditions), 5), dtype=np.float32)
    for idx in range(len(keyboard_conditions) - 1):
        poses[idx + 1] = compute_next_pose_from_action(poses[idx], keyboard_conditions[idx], mouse_conditions[idx])
    return poses


def build_intrinsics(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fov_rad = float(np.deg2rad(90.0))
    fx = float(width) / (2.0 * float(np.tan(fov_rad / 2.0)))
    fy = float(height) / (2.0 * float(np.tan(fov_rad / 2.0)))
    return torch.tensor([[fx, fy, width / 2.0, height / 2.0]], device=device, dtype=dtype)


def build_matrixgame3_action_preset(num_frames: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    presets = create_action_presets(num_frames=num_frames, keyboard_dim=6, seed=seed)
    return presets["keyboard"], presets["mouse"]


def interpolate_camera_poses_handedness(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    dets = np.linalg.det(src_rot_mat)
    flip_handedness = dets.size > 0 and np.median(dets) < 0.0
    if flip_handedness:
        flip_mat = np.diag([1.0, 1.0, -1.0]).astype(src_rot_mat.dtype)
        src_rot_mat = src_rot_mat @ flip_mat

    c2ws = interpolate_camera_poses(
        src_indices=src_indices,
        src_rot_mat=src_rot_mat,
        src_trans_vec=src_trans_vec,
        tgt_indices=tgt_indices,
    )
    if flip_handedness:
        flip_mat_t = torch.from_numpy(flip_mat).to(c2ws.device, dtype=c2ws.dtype)
        c2ws[:, :3, :3] = c2ws[:, :3, :3] @ flip_mat_t
    return c2ws


def build_extrinsics_from_actions(
    keyboard_conditions: torch.Tensor,
    mouse_conditions: torch.Tensor,
) -> torch.Tensor:
    keyboard_np = keyboard_conditions.detach().to(dtype=torch.float32).cpu().numpy()
    mouse_np = mouse_conditions.detach().to(dtype=torch.float32).cpu().numpy()
    poses = compute_all_poses_from_actions(keyboard_np, mouse_np)
    rotations = np.concatenate(
        [np.zeros((poses.shape[0], 1), dtype=np.float32), poses[:, 3:5]],
        axis=1,
    )
    positions = poses[:, :3]
    return build_extrinsics(rotations, positions)


def build_extrinsics(
    video_rotation: np.ndarray,
    video_position: np.ndarray,
) -> torch.Tensor:
    extrinsics = []
    for frame_rotation, frame_position in zip(video_rotation, video_position, strict=False):
        roll_deg, pitch_deg, yaw_deg = frame_rotation
        roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])

        rot_z = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        rot_y = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]],
            dtype=np.float32,
        )
        rot_x = np.array(
            [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
            dtype=np.float32,
        )
        rot = rot_z @ rot_y @ rot_x
        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = rot
        ext[:3, 3] = np.asarray(frame_position, dtype=np.float32)
        extrinsics.append(ext)

    r_init = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=np.float32)
    extrinsics_tensor = torch.from_numpy(np.stack(extrinsics, axis=0)).float()
    extrinsics_tensor[:, :3, :3] = extrinsics_tensor[:, :3, :3] @ torch.from_numpy(r_init)
    extrinsics_tensor[:, :3, 3] = extrinsics_tensor[:, :3, 3] * 0.01
    return extrinsics_tensor


def build_plucker_from_c2ws(
    c2ws_seq: torch.Tensor,
    src_indices: np.ndarray,
    tgt_indices: np.ndarray,
    *,
    target_h: int,
    target_w: int,
    latent_h: int,
    latent_w: int,
    framewise: bool = True,
) -> torch.Tensor:
    c2ws_np = c2ws_seq.detach().to(dtype=torch.float32).cpu().numpy()
    c2ws_infer = interpolate_camera_poses_handedness(
        src_indices=src_indices,
        src_rot_mat=c2ws_np[:, :3, :3],
        src_trans_vec=c2ws_np[:, :3, 3],
        tgt_indices=tgt_indices,
    ).to(device=c2ws_seq.device, dtype=c2ws_seq.dtype)
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=framewise)
    return build_plucker_from_pose(
        c2ws_infer,
        target_h=target_h,
        target_w=target_w,
        latent_h=latent_h,
        latent_w=latent_w,
    )


def build_plucker_from_pose(
    c2ws_pose: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    ks = build_intrinsics(target_h, target_w, c2ws_pose.device, c2ws_pose.dtype).repeat(c2ws_pose.shape[0], 1)
    plucker = get_plucker_embeddings(c2ws_pose, ks, target_h, target_w)
    c1 = target_h // latent_h
    c2 = target_w // latent_w
    plucker = plucker.view(c2ws_pose.shape[0], latent_h, c1, latent_w, c2, 6)
    plucker = plucker.permute(0, 1, 3, 5, 2, 4).reshape(c2ws_pose.shape[0], latent_h, latent_w, 6 * c1 * c2)
    plucker = plucker.permute(3, 0, 1, 2).unsqueeze(0)
    return plucker


def select_memory_idx_fov(
    extrinsics_all: torch.Tensor | np.ndarray,
    current_start_frame_idx: int,
    selected_index_base: list[int],
    *,
    height: int = 720,
    width: int = 1280,
    return_confidence: bool = False,
) -> list[int] | tuple[list[int], list[float]]:
    if isinstance(extrinsics_all, np.ndarray):
        extrinsics_tensor = torch.from_numpy(extrinsics_all).float()
    else:
        extrinsics_tensor = extrinsics_all.float()

    device = extrinsics_tensor.device
    if current_start_frame_idx <= 1:
        selected = [0] * len(selected_index_base)
        if return_confidence:
            return selected, [0.0] * len(selected)
        return selected

    fov_rad = np.deg2rad(90.0)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = height / (2 * np.tan(fov_rad / 2))
    near, far = 0.1, 30.0

    candidate_indices = torch.arange(1, current_start_frame_idx, device=device)
    r_cand = extrinsics_tensor[candidate_indices, :3, :3]
    t_cand = extrinsics_tensor[candidate_indices, :3, 3:4]
    r_cand_inv = r_cand.transpose(1, 2)
    t_cand_inv = -torch.bmm(r_cand_inv, t_cand)

    num_side = 10
    z_samples = torch.linspace(near, far, num_side, device=device)
    x_samples = torch.linspace(-1, 1, num_side, device=device)
    y_samples = torch.linspace(-1, 1, num_side, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x_samples, y_samples, z_samples, indexing="ij")
    points_cam_base = torch.stack([
        grid_x.reshape(-1) * grid_z.reshape(-1) * (width / (2 * fx)),
        grid_y.reshape(-1) * grid_z.reshape(-1) * (height / (2 * fy)),
        grid_z.reshape(-1),
    ], dim=0)

    selected_index: list[int] = []
    selected_confidence: list[float] = []
    for frame_idx in selected_index_base:
        base_pose = extrinsics_tensor[frame_idx]
        points_world = base_pose[:3, :3] @ points_cam_base + base_pose[:3, 3:4]
        points_world_batched = points_world.unsqueeze(0)
        points_in_cands = torch.bmm(r_cand_inv, points_world_batched.expand(len(candidate_indices), -1, -1)) + t_cand_inv

        x = points_in_cands[:, 0, :]
        y = points_in_cands[:, 1, :]
        z = points_in_cands[:, 2, :]
        u = (x * fx / torch.clamp(z, min=1e-6)) + width / 2
        v = (y * fy / torch.clamp(z, min=1e-6)) + height / 2

        in_view = (z > near) & (z < far) & (u >= 0) & (u <= width) & (v >= 0) & (v <= height)
        ratios = in_view.float().mean(dim=1)
        best_idx = torch.argmax(ratios)
        selected_index.append(candidate_indices[best_idx].item())
        selected_confidence.append(ratios[best_idx].item())

    if return_confidence:
        return selected_index, selected_confidence
    return selected_index


def create_action_presets(num_frames: int, keyboard_dim: int = 4, seed: int = None):
    if keyboard_dim not in (2, 4, 6, 7):
        raise ValueError(f"keyboard_dim must be 2, 4, 6, or 7, got {keyboard_dim}")
    if num_frames % 4 != 1:
        raise ValueError("Matrix-Game conditioning expects num_frames to be 4k+1.")

    if seed is not None:
        random.seed(seed)

    num_samples_per_action = 4

    if keyboard_dim == 4:
        actions_single_action = ["forward", "left", "right"]
        actions_double_action = ["forward_left", "forward_right"]
        actions_single_camera = ["camera_l", "camera_r"]
        keyboard_idx = {"forward": 0, "back": 1, "left": 2, "right": 3}
    elif keyboard_dim == 2:
        actions_single_action = ["forward", "back"]
        actions_double_action = []
        actions_single_camera = ["camera_l", "camera_r"]
        keyboard_idx = {"forward": 0, "back": 1}
    elif keyboard_dim == 6:
        actions_single_action = ["forward", "back", "left", "right"]
        actions_double_action = ["forward_left", "forward_right"]
        actions_single_camera = ["camera_l", "camera_r"]
        keyboard_idx = {"forward": 0, "back": 1, "left": 2, "right": 3, "t1": 4, "t2": 5}
    else:  # keyboard_dim == 7
        actions_single_action = ["forward", "back", "left", "right"]
        actions_double_action = []
        actions_single_camera = []
        keyboard_idx = {"still": 0, "forward": 1, "back": 2, "left": 3, "right": 4, "a": 5, "d": 6}

    actions_to_test = (
        actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    )
    for action in (actions_single_action + actions_double_action):
        for camera in actions_single_camera:
            actions_to_test.append(f"{action}_{camera}")

    if not actions_to_test:
        actions_to_test = actions_single_action * 5

    base_action = actions_single_action + actions_single_camera
    cam_value = 0.1
    camera_value_map = {
        "camera_up": [cam_value, 0],
        "camera_down": [-cam_value, 0],
        "camera_l": [0, -cam_value],
        "camera_r": [0, cam_value],
        "camera_ur": [cam_value, cam_value],
        "camera_ul": [cam_value, -cam_value],
        "camera_dr": [-cam_value, cam_value],
        "camera_dl": [-cam_value, -cam_value],
    }

    data = []
    for action_name in actions_to_test:
        keyboard_condition = torch.zeros((num_samples_per_action, keyboard_dim))
        mouse_condition = torch.zeros((num_samples_per_action, 2))

        for sub_act in base_action:
            if sub_act not in action_name:
                continue
            if sub_act in camera_value_map:
                mouse_condition = torch.tensor(
                    [camera_value_map[sub_act] for _ in range(num_samples_per_action)],
                    dtype=mouse_condition.dtype,
                )
            elif sub_act in keyboard_idx:
                keyboard_condition[:, keyboard_idx[sub_act]] = 1

        data.append({
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition,
        })

    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    mouse_condition = torch.zeros((num_frames, 2))
    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        entry = data[random.randint(0, len(data) - 1)]
        key_seq = entry["keyboard_condition"]
        mouse_seq = entry["mouse_condition"]

        if current_frame == 0:
            keyboard_condition[:1] = key_seq[:1]
            mouse_condition[:1] = mouse_seq[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            keyboard_condition[current_frame:current_frame + rd_frame] = key_seq.repeat(repeat_time, 1)
            mouse_condition[current_frame:current_frame + rd_frame] = mouse_seq.repeat(repeat_time, 1)
            current_frame += rd_frame

    return {"keyboard": keyboard_condition, "mouse": mouse_condition}
