from __future__ import annotations

import asyncio
import os
import random

import numpy as np
import torch
from PIL import Image

from fastvideo.distributed.parallel_state import get_local_torch_device
from fastvideo.models.dits.lingbotworld.cam_utils import (
    compute_relative_poses,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from fastvideo.utils import logger


CAM_VALUE = 0.1
CAMERA_MAP = {
    "i": [CAM_VALUE, 0], "k": [-CAM_VALUE, 0],
    "j": [0, -CAM_VALUE], "l": [0, CAM_VALUE], "u": [0, 0]
}

KEYBOARD_MAP_4 = {  # base_distilled_model (universal): W/S/A/D
    "w": [1, 0, 0, 0], "s": [0, 1, 0, 0],
    "a": [0, 0, 1, 0], "d": [0, 0, 0, 1], "q": [0, 0, 0, 0]
}
KEYBOARD_MAP_2 = {  # gta_distilled_model: W/S only (steering via mouse)
    "w": [1, 0], "s": [0, 1], "q": [0, 0]
}
KEYBOARD_MAP_7 = {  # templerun_distilled_model: still/w/s/left/right/a/d
    "q": [1, 0, 0, 0, 0, 0, 0],  # still
    "w": [0, 1, 0, 0, 0, 0, 0],  # forward
    "s": [0, 0, 1, 0, 0, 0, 0],  # back
    "j": [0, 0, 0, 1, 0, 0, 0],  # left (swipe)
    "l": [0, 0, 0, 0, 1, 0, 0],  # right (swipe)
    "a": [0, 0, 0, 0, 0, 1, 0],  # a
    "d": [0, 0, 0, 0, 0, 0, 1],  # d
}
KEYBOARD_MAP = KEYBOARD_MAP_4  # Default for backward compatibility

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

    while new_yaw > 180:
        new_yaw -= 360
    while new_yaw < -180:
        new_yaw += 360

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


def pose_to_c2w(pose):
    x, y, z, pitch_deg, yaw_deg = pose
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    rot_x = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]],
                     dtype=np.float32)
    rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], dtype=np.float32)
    rot = rot_z @ rot_x
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return c2w


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


def build_plucker_from_actions(
    keyboard_window: torch.Tensor,
    mouse_window: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    c2ws = build_extrinsics_from_actions(keyboard_window, mouse_window)
    relative = compute_relative_poses(c2ws, framewise=True, normalize_trans=True)
    return build_plucker_from_pose(
        relative,
        target_h=target_h,
        target_w=target_w,
        latent_h=latent_h,
        latent_w=latent_w,
    )

def expand_action_to_frames(action: dict, num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    result = {}
    for key, tensor in action.items():
        if tensor is not None:
             # Expand to [num_frames, D] then unsqueeze to [1, num_frames, D]
            result[key] = tensor.unsqueeze(0).repeat(num_frames, 1).unsqueeze(0)
        else:
            result[key] = None

    if "mouse" not in result or result["mouse"] is None:
         # keyboard device if available, otherwise default
         device = result.get("keyboard", torch.tensor([])).device if result.get("keyboard") is not None else get_local_torch_device()
         result["mouse"] = torch.zeros(1, num_frames, 2, device=device)
    
    return result["keyboard"], result["mouse"]

def get_current_action(mode="universal"):

    CAM_VALUE = 0.1
    if mode == 'universal':
        logger.info("")
        logger.info('-'*30)
        logger.info("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM\n (I: up, K: down, J: left, L: right, U: no move)")
        logger.info("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        logger.info('-'*30)
        CAMERA_VALUE_MAP = {
            "i":  [CAM_VALUE, 0],
            "k":  [-CAM_VALUE, 0],
            "j":  [0, -CAM_VALUE],
            "l":  [0, CAM_VALUE],
            "u":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1],
            "q": [0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_mouse = input('Please input the mouse action (e.g. `U`):\n').strip().lower()
                idx_keyboard = input('Please input the keyboard action (e.g. `W`):\n').strip().lower()
                if idx_mouse in CAMERA_VALUE_MAP and idx_keyboard in KEYBOARD_IDX:
                    flag = 1
            except Exception:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    elif mode == 'gta_drive':
        logger.info("")
        logger.info('-'*30)
        logger.info("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        logger.info('-'*30)
        CAMERA_VALUE_MAP = {
            "a":  [0, -CAM_VALUE],
            "d":  [0, CAM_VALUE],
            "q":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0], "s": [0, 1],
            "q": [0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                indexes = input('Please input the actions (split with ` `):\n(e.g. `W` for forward, `W A` for forward and left)\n').strip().lower().split(' ')
                idx_mouse = []
                idx_keyboard = []
                for i in indexes:
                    if i in CAMERA_VALUE_MAP:
                        idx_mouse += [i]
                    elif i in KEYBOARD_IDX:
                        idx_keyboard += [i]
                if len(idx_mouse) == 0:
                    idx_mouse += ['q']
                if len(idx_keyboard) == 0:
                    idx_keyboard += ['q']
                assert idx_mouse in [['a'], ['d'], ['q']] and idx_keyboard in [['q'], ['w'], ['s']]
                flag = 1
            except Exception:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse[0]]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard[0]]).cuda()
    elif mode == 'templerun':
        logger.info("")
        logger.info('-'*30)
        logger.info("PRESS [W, S, A, D, Z, C, Q] FOR ACTIONS\n (W: jump, S: slide, A: left side, D: right side, Z: turn left, C: turn right, Q: no move)")
        logger.info('-'*30)
        KEYBOARD_IDX = { 
            "w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
            "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
            "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
            "q": [1, 0, 0, 0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_keyboard = input('Please input the action: \n(e.g. `W` for forward, `Z` for turning left)\n').strip().lower()
                if idx_keyboard in KEYBOARD_IDX:
                    flag = 1
            except Exception:
                pass
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    
    if mode != 'templerun':
        return {
            "mouse": mouse_cond,
            "keyboard": keyboard_cond
        }
    return {
        "keyboard": keyboard_cond
    }

async def get_current_action_async(mode="universal"):
    return await asyncio.to_thread(get_current_action, mode)

def load_initial_image(image_path: str = None) -> Image.Image:
    if image_path and os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")
    logger.warning("No image provided, creating placeholder...")
    return Image.new("RGB", (640, 352), (128, 128, 128))

def create_action_presets(num_frames: int, keyboard_dim: int = 4, seed: int = None):
    if keyboard_dim not in (2, 4, 6, 7):
        raise ValueError(f"keyboard_dim must be 2, 4, 6, or 7, got {keyboard_dim}")
    if num_frames % 4 != 1:
        raise ValueError("Matrix-Game conditioning expects num_frames to be 4k+1.")

    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    num_samples_per_action = 4
    
    # Define actions based on keyboard_dim
    if keyboard_dim == 4:
        # Universal model: W, S, A, D
        actions_single_action = ["forward", "left", "right"]
        actions_double_action = ["forward_left", "forward_right"]
        actions_single_camera = ["camera_l", "camera_r"]
        keyboard_idx = {"forward": 0, "back": 1, "left": 2, "right": 3}
    elif keyboard_dim == 2:
        # GTA model: W, S only (steering via mouse)
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
        # Temple Run model: still, w, s, left, right, a, d (no mouse)
        actions_single_action = ["forward", "back", "left", "right"]
        actions_double_action = []
        actions_single_camera = []  # No mouse for Temple Run
        keyboard_idx = {"still": 0, "forward": 1, "back": 2, "left": 3, "right": 4, "a": 5, "d": 6}

    actions_to_test = (
        actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    )
    for action in (actions_single_action + actions_double_action):
        for camera in actions_single_camera:
            actions_to_test.append(f"{action}_{camera}")

    # Ensure we have at least some actions
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

def parse_config(config, mode="universal"):
    assert mode in ['universal', 'gta_drive', 'templerun']
    key_data = {}
    mouse_data = {}
    if mode != 'templerun':
        key, mouse = config
    else:
        key = config

    for i in range(len(key)):
        if mode == 'templerun':
            still, w, s, left, right, a, d = key[i]
        elif mode == 'universal':
            w, s, a, d = key[i]
        else:
            w, s, a, d = key[i][0], key[i][1], mouse[i][1] < 0, mouse[i][1] > 0
        if mode == 'universal':
            mouse_y, mouse_x = mouse[i]
            mouse_y = -1 * mouse_y

        key_data[i] = {"W": bool(w), "A": bool(a), "S": bool(s), "D": bool(d)}
        if mode == 'templerun':
            key_data[i].update({"left": bool(left), "right": bool(right)})

        if mode == 'universal':
            if i == 0:
                mouse_data[i] = (320, 352 // 2)
            else:
                global_scale_factor = 0.1
                mouse_scale_x = 15 * global_scale_factor
                mouse_scale_y = 15 * 4 * global_scale_factor
                mouse_data[i] = (
                    mouse_data[i - 1][0] + mouse_x * mouse_scale_x,
                    mouse_data[i - 1][1] + mouse_y * mouse_scale_y,
                )
    return key_data, mouse_data

# NOTE: drawing functions are commented out to avoid cv2/libGL dependency.
#
# def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, alpha=0.5):
#     overlay = image.copy()
#     x1, y1 = top_left
#     x2, y2 = bottom_right
#
#     cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
#     cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
#     cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
#     cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
#     cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
#     cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
#     cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
#
# def draw_keys_on_frame(frame, keys, key_size=(80, 50), spacing=20, bottom_margin=30, mode='universal'):
#     h, w, _ = frame.shape
#     horison_shift = 90
#     vertical_shift = -20
#     horizon_shift_all = 50
#     key_positions = {
#         "W": (w // 2 - key_size[0] // 2 - horison_shift - horizon_shift_all, 
#               h - bottom_margin - key_size[1] * 2 + vertical_shift - 20),
#         "A": (w // 2 - key_size[0] * 2 + 5 - horison_shift - horizon_shift_all, 
#               h - bottom_margin - key_size[1] + vertical_shift),
#         "S": (w // 2 - key_size[0] // 2 - horison_shift - horizon_shift_all, 
#               h - bottom_margin - key_size[1] + vertical_shift),
#         "D": (w // 2 + key_size[0] - 5 - horison_shift - horizon_shift_all, 
#               h - bottom_margin - key_size[1] + vertical_shift),
#     }
#     key_icon = {"W": "W", "A": "A", "S": "S", "D": "D", "left": "left", "right": "right"}
#     if mode == 'templerun':
#         key_positions.update({
#             "left": (w // 2 + key_size[0] * 2 + spacing * 2 - horison_shift - horizon_shift_all, 
#                      h - bottom_margin - key_size[1] + vertical_shift),
#             "right": (w // 2 + key_size[0] * 3 + spacing * 7 - horison_shift - horizon_shift_all, 
#                       h - bottom_margin - key_size[1] + vertical_shift)
#         })
#
#     for key, (x, y) in key_positions.items():
#         is_pressed = keys.get(key, False)
#         top_left = (x, y)
#         if key in ["left", "right"]:
#             bottom_right = (x + key_size[0] + 40, y + key_size[1])
#         else:
#             bottom_right = (x + key_size[0], y + key_size[1])
#
#         color = (0, 255, 0) if is_pressed else (200, 200, 200)
#         alpha = 0.8 if is_pressed else 0.5
#         draw_rounded_rectangle(frame, top_left, bottom_right, color, radius=10, alpha=alpha)
#
#         text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
#         if key in ["left", "right"]:
#             text_x = x + (key_size[0] + 40 - text_size[0]) // 2
#         else:
#             text_x = x + (key_size[0] - text_size[0]) // 2
#         text_y = y + (key_size[1] + text_size[1]) // 2
#         cv2.putText(frame, key_icon[key], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#
# def overlay_icon(frame, icon, position, scale=1.0, rotation=0):
#     x, y = position
#     h, w, _ = icon.shape
#
#     scaled_width = int(w * scale)
#     scaled_height = int(h * scale)
#     icon_resized = cv2.resize(icon, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
#
#     center = (scaled_width // 2, scaled_height // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
#     icon_rotated = cv2.warpAffine(
#         icon_resized, rotation_matrix, (scaled_width, scaled_height),
#         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
#     )
#
#     h, w, _ = icon_rotated.shape
#     frame_h, frame_w, _ = frame.shape
#
#     top_left_x = max(0, int(x - w // 2))
#     top_left_y = max(0, int(y - h // 2))
#     bottom_right_x = min(frame_w, int(x + w // 2))
#     bottom_right_y = min(frame_h, int(y + h // 2))
#
#     icon_x_start = max(0, int(-x + w // 2))
#     icon_y_start = max(0, int(-y + h // 2))
#     icon_x_end = icon_x_start + (bottom_right_x - top_left_x)
#     icon_y_end = icon_y_start + (bottom_right_y - top_left_y)
#
#     icon_region = icon_rotated[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
#     alpha = icon_region[:, :, 3] / 255.0
#     icon_rgb = icon_region[:, :, :3]
#
#     frame_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
#     for c in range(3):
#         frame_region[:, :, c] = (1 - alpha) * frame_region[:, :, c] + alpha * icon_rgb[:, :, c]
#     frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame_region
#
# def process_video(input_video, output_video, config, mouse_icon_path, 
#                   mouse_scale=1.0, mouse_rotation=0, process_icon=True, mode='universal'):
#     key_data, mouse_data = parse_config(config, mode=mode)
#     fps = 12
#
#     mouse_icon = cv2.imread(mouse_icon_path, cv2.IMREAD_UNCHANGED)
#
#     out_video = []
#     for frame_idx, frame in enumerate(input_video):
#         frame = np.ascontiguousarray(frame)
#         if process_icon:
#             keys = key_data.get(frame_idx, {"W": False, "A": False, "S": False, "D": False, "left": False, "right": False})
#             draw_keys_on_frame(frame, keys, key_size=(50, 50), spacing=10, bottom_margin=20, mode=mode)
#             if mode == 'universal':
#                 frame_width = frame.shape[1]
#                 frame_height = frame.shape[0]
#                 mouse_position = mouse_data.get(frame_idx, (frame_width // 2, frame_height // 2))
#                 overlay_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
#         out_video.append(frame / 255)
#     
#     export_to_video(out_video, output_video, fps=fps)
#     logger.info(f"Video saved to {output_video}")
