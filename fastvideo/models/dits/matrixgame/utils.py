from __future__ import annotations

import asyncio
import os
import random

import torch
from PIL import Image

from fastvideo.distributed.parallel_state import get_local_torch_device
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
                    if i in CAMERA_VALUE_MAP.keys():
                        idx_mouse += [i]
                    elif i in KEYBOARD_IDX.keys():
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
                if idx_keyboard in KEYBOARD_IDX.keys():
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
import cv2
import numpy as np
from diffusers.utils import export_to_video

def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, alpha=0.5):
    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def draw_keys_on_frame(frame, keys, key_size=(30, 30), spacing=5, top_margin=15, mode='universal'):
    """Draw WASD keys on the left top of the frame."""
    h, w, _ = frame.shape
    
    # Left top positioning
    left_margin = 15
    gap = 3  # Gap between keys
    
    key_positions = {
        "W": (left_margin + key_size[0] + gap, 
              top_margin),
        "A": (left_margin, 
              top_margin + key_size[1] + gap),
        "S": (left_margin + key_size[0] + gap, 
              top_margin + key_size[1] + gap),
        "D": (left_margin + (key_size[0] + gap) * 2, 
              top_margin + key_size[1] + gap),
    }
    key_icon = {"W": "W", "A": "A", "S": "S", "D": "D", "left": "L", "right": "R"}
    if mode == 'templerun':
        key_positions.update({
            "left": (left_margin + (key_size[0] + gap) * 3 + 10, 
                     top_margin + key_size[1] + gap),
            "right": (left_margin + (key_size[0] + gap) * 4 + 15, 
                      top_margin + key_size[1] + gap)
        })

    for key, (x, y) in key_positions.items():
        is_pressed = keys.get(key, False)
        top_left = (x, y)
        bottom_right = (x + key_size[0], y + key_size[1])

        color = (0, 255, 0) if is_pressed else (200, 200, 200)
        alpha = 0.8 if is_pressed else 0.5
        draw_rounded_rectangle(frame, top_left, bottom_right, color, radius=5, alpha=alpha)

        text_size = cv2.getTextSize(key_icon[key], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (key_size[0] - text_size[0]) // 2
        text_y = y + (key_size[1] + text_size[1]) // 2
        cv2.putText(frame, key_icon[key], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def overlay_icon(frame, icon, position, scale=1.0, rotation=0):
    x, y = position
    h, w, _ = icon.shape

    scaled_width = int(w * scale)
    scaled_height = int(h * scale)
    icon_resized = cv2.resize(icon, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    center = (scaled_width // 2, scaled_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    icon_rotated = cv2.warpAffine(
        icon_resized, rotation_matrix, (scaled_width, scaled_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

    h, w, _ = icon_rotated.shape
    frame_h, frame_w, _ = frame.shape

    top_left_x = max(0, int(x - w // 2))
    top_left_y = max(0, int(y - h // 2))
    bottom_right_x = min(frame_w, int(x + w // 2))
    bottom_right_y = min(frame_h, int(y + h // 2))

    icon_x_start = max(0, int(-x + w // 2))
    icon_y_start = max(0, int(-y + h // 2))
    icon_x_end = icon_x_start + (bottom_right_x - top_left_x)
    icon_y_end = icon_y_start + (bottom_right_y - top_left_y)

    icon_region = icon_rotated[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
    alpha = icon_region[:, :, 3] / 255.0
    icon_rgb = icon_region[:, :, :3]

    frame_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    for c in range(3):
        frame_region[:, :, c] = (1 - alpha) * frame_region[:, :, c] + alpha * icon_rgb[:, :, c]
    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame_region

def process_video(input_video, output_video, config, mouse_icon_path, 
                  mouse_scale=1.0, mouse_rotation=0, process_icon=True, mode='universal'):
    key_data, mouse_data = parse_config(config, mode=mode)
    fps = 12

    mouse_icon = cv2.imread(mouse_icon_path, cv2.IMREAD_UNCHANGED)

    out_video = []
    for frame_idx, frame in enumerate(input_video):
        frame = np.ascontiguousarray(frame)
        if process_icon:
            keys = key_data.get(frame_idx, {"W": False, "A": False, "S": False, "D": False, "left": False, "right": False})
            draw_keys_on_frame(frame, keys, key_size=(50, 50), spacing=10, bottom_margin=20, mode=mode)
            if mode == 'universal':
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                mouse_position = mouse_data.get(frame_idx, (frame_width // 2, frame_height // 2))
                overlay_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
        out_video.append(frame / 255)
    
    export_to_video(out_video, output_video, fps=fps)
    logger.info(f"Video saved to {output_video}")


def parse_npy_action(action_path):
    """Convert npy action file to key_data and mouse_data dict format."""
    action_data = np.load(action_path, allow_pickle=True).item()
    keyboard_data = action_data['keyboard']  # shape: (num_frames, 6) -> [W, S, A, D, left, right]
    mouse_data = action_data.get('mouse', None)  # shape: (num_frames, 2) -> [Pitch, Yaw]
    
    # MatrixGame convention: 0:W, 1:S, 2:A, 3:D, 4:left, 5:right
    key_names = ["W", "S", "A", "D", "left", "right"]
    key_data = {}
    for frame_idx, keys in enumerate(keyboard_data):
        key_data[frame_idx] = {key_names[i]: bool(keys[i]) for i in range(len(key_names))}
    
    # MatrixGame convention: mouse is [Pitch, Yaw]
    mouse_dict = {}
    if mouse_data is not None:
        for frame_idx, (pitch, yaw) in enumerate(mouse_data):
            mouse_dict[frame_idx] = {"pitch": float(pitch), "yaw": float(yaw)}
    
    return key_data, mouse_dict


def draw_mouse_on_frame(frame, pitch, yaw, top_margin=15):
    """Draw crosshair with direction arrow on the right top of the frame."""
    h, w, _ = frame.shape
    
    # Right top positioning
    right_margin = 15
    crosshair_radius = 25
    
    # Position crosshair on the right top
    crosshair_x = w - right_margin - crosshair_radius
    crosshair_y = top_margin + crosshair_radius
    
    # Yaw affects horizontal direction, pitch affects vertical
    dx = int(yaw * crosshair_radius * 8)  # Scale for visibility
    dy = int(-pitch * crosshair_radius * 8)  # Negative because y increases downward
    
    # Clamp arrow length
    max_arrow = crosshair_radius - 5
    dx = max(-max_arrow, min(max_arrow, dx))
    dy = max(-max_arrow, min(max_arrow, dy))
    
    # Draw crosshair background
    cv2.circle(frame, (crosshair_x, crosshair_y), crosshair_radius, (50, 50, 50), -1)
    cv2.circle(frame, (crosshair_x, crosshair_y), crosshair_radius, (200, 200, 200), 1)
    cv2.line(frame, (crosshair_x - crosshair_radius + 5, crosshair_y), 
             (crosshair_x + crosshair_radius - 5, crosshair_y), (100, 100, 100), 1)
    cv2.line(frame, (crosshair_x, crosshair_y - crosshair_radius + 5), 
             (crosshair_x, crosshair_y + crosshair_radius - 5), (100, 100, 100), 1)
    
    # Draw direction arrow
    if abs(dx) > 1 or abs(dy) > 1:
        cv2.arrowedLine(frame, (crosshair_x, crosshair_y), (crosshair_x + dx, crosshair_y + dy), 
                       (0, 255, 0), 2, tipLength=0.3)


def process_video_with_npy(input_video, output_video, action_path, fps=12, mode='universal'):
    """Process video with overlay using npy action file.
    
    Uses existing draw_keys_on_frame function.
    """
    key_data, mouse_data = parse_npy_action(action_path)
    
    out_video = []
    for frame_idx, frame in enumerate(input_video):
        frame = np.ascontiguousarray(frame)
        keys = key_data.get(frame_idx, {"W": False, "A": False, "S": False, "D": False, "left": False, "right": False})
        draw_keys_on_frame(frame, keys, mode=mode)
        
        # Draw pitch and yaw
        mouse = mouse_data.get(frame_idx, {"pitch": 0.0, "yaw": 0.0})
        draw_mouse_on_frame(frame, mouse["pitch"], mouse["yaw"])
        
        out_video.append(frame / 255.0)
    
    export_to_video(out_video, output_video, fps=fps)
    logger.info(f"Video saved to {output_video}")


if __name__ == "__main__":
    import argparse
    import cv2
    
    parser = argparse.ArgumentParser(description="Overlay keyboard actions on video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4)")
    parser.add_argument("--action", type=str, required=True, help="Path to action file (.npy)")
    parser.add_argument("--output", type=str, default=None, help="Path to output video (default: input_with_overlay.mp4)")
    parser.add_argument("--fps", type=int, default=12, help="Output video FPS")
    args = parser.parse_args()
    
    # Load video frames using cv2
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    print(f"Loaded {len(frames)} frames from video")
    
    # Set output path
    if args.output is None:
        base_name = args.video.rsplit('.', 1)[0]
        output_path = f"{base_name}_with_overlay.mp4"
    else:
        output_path = args.output
    
    # Process video with overlay using existing functions
    process_video_with_npy(frames, output_path, args.action, fps=args.fps)
    print(f"Video with overlay saved to: {output_path}")
