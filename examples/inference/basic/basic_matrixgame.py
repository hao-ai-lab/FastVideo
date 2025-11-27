from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import MatrixGameI2V480PConfig
from fastvideo.models.dits.matrix_game.causal_model import CausalMatrixGameWanModel

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

OUTPUT_PATH = "outputs/matrixgame"


def create_action_presets(num_frames: int, keyboard_dim: int = 4):
    if keyboard_dim != 4:
        raise ValueError("This demo only supports the universal (4-keyboard) Matrix-Game model.")
    if num_frames % 4 != 1:
        raise ValueError("Matrix-Game conditioning expects num_frames to be 4k+1.")

    num_samples_per_action = 4
    actions_single_action = ["forward", "left", "right"]
    actions_double_action = ["forward_left", "forward_right"]
    actions_single_camera = ["camera_l", "camera_r"]
    actions_to_test = (
        actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    )
    for action in (actions_single_action + actions_double_action):
        for camera in actions_single_camera:
            actions_to_test.append(f"{action}_{camera}")

    base_action = actions_single_action + actions_single_camera
    keyboard_idx = {"forward": 0, "back": 1, "left": 2, "right": 3}
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

    # Build the per-action snippets
    data = []
    for action_name in actions_to_test:
        keyboard_condition = torch.zeros((num_samples_per_action, 4))
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

        data.append(
            {
                "keyboard_condition": keyboard_condition,
                "mouse_condition": mouse_condition,
            }
        )

    # Combine the snippets into the frame-by-frame conditioning
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


def load_initial_image(image_path: str = None) -> Image.Image:
    if image_path and os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")
    print("No image provided, creating placeholder...")
    return Image.new("RGB", (640, 352), color=(128, 128, 128))


def main():
    # model_path = "/workspace/Matrix-Game-2.0-Diffusers/base_model"
    model_path = "/workspace/Matrix-Game-2.0-Diffusers/base_distilled_model"
    image_path = "/FastVideo/Matrix-Game/Matrix-Game-2/demo_images/universal/0002.png"
    num_output_frames = 150
    num_frames = (num_output_frames - 1) * 4 + 1  # 4k+1 actual video frames
    seed = 42

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    actions = create_action_presets(num_frames, keyboard_dim=4)
    keyboard_cond = actions["keyboard"]
    mouse_cond = actions["mouse"]

    initial_image = load_initial_image(image_path)

    # FastVideo will automatically use the optimal default arguments for the model
    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        pipeline_config=MatrixGameI2V480PConfig()
    )

    latent_h = 352 // 8
    latent_w = 640 // 8
    latent_f = (num_frames - 1) // 4 + 1
    grid_sizes = torch.tensor([latent_f, latent_h, latent_w])

    output_path = Path(OUTPUT_PATH) / "output.mp4"
    video = generator.generate_video(
        prompt="",
        pil_image=initial_image,
        mouse_cond=mouse_cond.unsqueeze(0),
        keyboard_cond=keyboard_cond.unsqueeze(0),
        grid_sizes=grid_sizes,
        num_frames=num_frames,
        height=352,
        width=640,
        num_inference_steps=50,
        seed=seed,
        output_path=str(output_path),
        save_video=True,
    )

if __name__ == "__main__":
    main()
