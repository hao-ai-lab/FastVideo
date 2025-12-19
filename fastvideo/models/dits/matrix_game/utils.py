from __future__ import annotations

import os
import random

# import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

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


def load_initial_image(image_path: str = None) -> Image.Image:
    if image_path and os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")
    logger.warning("No image provided, creating placeholder...")
    return Image.new("RGB", (640, 352), (128, 128, 128))


def create_action_presets(num_frames: int, keyboard_dim: int = 4, seed: int = None):
    if keyboard_dim not in (2, 4, 7):
        raise ValueError(f"keyboard_dim must be 2, 4, or 7, got {keyboard_dim}")
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
