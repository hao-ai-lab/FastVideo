from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import MatrixGameI2V480PConfig
from fastvideo.models.dits.matrix_game.causal_model import CausalMatrixGameWanModel

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

OUTPUT_PATH = "outputs/matrixgame"


def create_action_presets(num_frames: int, keyboard_dim: int = 4) -> Dict[str, Dict[str, torch.Tensor]]:
    presets = {}

    if keyboard_dim == 4:  # Base model (forward, back, left, right)
        # Forward only
        keyboard = torch.zeros(num_frames, 4)
        keyboard[:, 0] = 1  # forward
        mouse = torch.zeros(num_frames, 2)
        presets["forward"] = {"keyboard": keyboard, "mouse": mouse}

        # Forward + turn right
        keyboard = torch.zeros(num_frames, 4)
        keyboard[:, 0] = 1  # forward
        keyboard[:, 3] = 1  # right
        mouse = torch.zeros(num_frames, 2)
        mouse[:, 1] = 0.1  # camera right
        presets["forward_right"] = {"keyboard": keyboard, "mouse": mouse}

        # Forward + turn left
        keyboard = torch.zeros(num_frames, 4)
        keyboard[:, 0] = 1  # forward
        keyboard[:, 2] = 1  # left
        mouse = torch.zeros(num_frames, 2)
        mouse[:, 1] = -0.1  # camera left
        presets["forward_left"] = {"keyboard": keyboard, "mouse": mouse}

        # Circle motion (forward + rotating camera)
        keyboard = torch.zeros(num_frames, 4)
        keyboard[:, 0] = 1  # forward
        mouse = torch.zeros(num_frames, 2)
        # Gradually rotate camera
        angle = torch.linspace(0, 2 * np.pi, num_frames)
        mouse[:, 0] = 0.1 * torch.sin(angle)  # y (vertical)
        mouse[:, 1] = 0.1 * torch.cos(angle)  # x (horizontal)
        mouse[:, 1] = 0.1 * torch.cos(angle)  # x (horizontal)
        presets["circle"] = {"keyboard": keyboard, "mouse": mouse}

    elif keyboard_dim == 6:  # Base model with 6 keys (WASD + 2 unknown/unused)
        # Forward only
        keyboard = torch.zeros(num_frames, 6)
        keyboard[:, 0] = 1  # forward
        mouse = torch.zeros(num_frames, 2)
        presets["forward"] = {"keyboard": keyboard, "mouse": mouse}

        # Forward + turn right
        keyboard = torch.zeros(num_frames, 6)
        keyboard[:, 0] = 1  # forward
        keyboard[:, 3] = 1  # right
        mouse = torch.zeros(num_frames, 2)
        mouse[:, 1] = 0.1  # camera right
        presets["forward_right"] = {"keyboard": keyboard, "mouse": mouse}

        # Forward + turn left
        keyboard = torch.zeros(num_frames, 6)
        keyboard[:, 0] = 1  # forward
        keyboard[:, 2] = 1  # left
        mouse = torch.zeros(num_frames, 2)
        mouse[:, 1] = -0.1  # camera left
        presets["forward_left"] = {"keyboard": keyboard, "mouse": mouse}

        # Circle motion (forward + rotating camera)
        keyboard = torch.zeros(num_frames, 6)
        keyboard[:, 0] = 1  # forward
        mouse = torch.zeros(num_frames, 2)
        # Gradually rotate camera
        angle = torch.linspace(0, 2 * np.pi, num_frames)
        mouse[:, 0] = 0.1 * torch.sin(angle)  # y (vertical)
        mouse[:, 1] = 0.1 * torch.cos(angle)  # x (horizontal)
        presets["circle"] = {"keyboard": keyboard, "mouse": mouse}

    elif keyboard_dim == 2:  # GTA model (forward, back)
        # Forward only
        keyboard = torch.zeros(num_frames, 2)
        keyboard[:, 0] = 1  # forward
        mouse = torch.zeros(num_frames, 2)
        presets["forward"] = {"keyboard": keyboard, "mouse": mouse}

        # Forward + turn right
        keyboard = torch.zeros(num_frames, 2)
        keyboard[:, 0] = 1  # forward
        mouse = torch.zeros(num_frames, 2)
        mouse[:, 1] = 0.1  # camera right
        presets["forward_right"] = {"keyboard": keyboard, "mouse": mouse}

    elif keyboard_dim == 7:  # Temple Run model (nomove, jump, slide, turnleft, turnright, leftside, rightside)
        # Jump sequence
        keyboard = torch.zeros(num_frames, 7)
        # Alternate between nomove and jump
        keyboard[::8, 0] = 1  # nomove
        keyboard[1::8, 1] = 1  # jump
        presets["jump"] = {"keyboard": keyboard}

        # Slide sequence
        keyboard = torch.zeros(num_frames, 7)
        keyboard[::8, 0] = 1  # nomove
        keyboard[1::8, 2] = 1  # slide
        presets["slide"] = {"keyboard": keyboard}

        # Turn left/right sequence
        keyboard = torch.zeros(num_frames, 7)
        keyboard[:num_frames//3, 3] = 1  # turn left
        keyboard[num_frames//3:2*num_frames//3, 0] = 1  # nomove
        keyboard[2*num_frames//3:, 4] = 1  # turn right
        presets["turn_sequence"] = {"keyboard": keyboard}

    return presets


def load_initial_image(image_path: str = None) -> Image.Image:
    if image_path and os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")
    print("No image provided, creating placeholder...")
    return Image.new("RGB", (640, 352), color=(128, 128, 128))


def main():
    # model_path = "/workspace/Matrix-Game-2.0-Diffusers/base_model"
    model_path = "/workspace/Matrix-Game-2.0-Diffusers/base_distilled_model"
    image_path = "/FastVideo/Matrix-Game/Matrix-Game-2/demo_images/universal/0002.png"
    action_preset = "forward_right"
    num_frames = 57  # Must be 4k+1
    seed = 42

    if (num_frames - 1) % 4 != 0:
        raise ValueError(f"num_frames must be 4k+1, got {num_frames}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model config to determine keyboard_dim
    config_path = Path(model_path) / "transformer" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path) as f:
        model_config = json.load(f)

    action_config = model_config.get("action_config")
    if not action_config:
        raise ValueError("Model does not have action_config. Is this a Matrix-Game model?")

    keyboard_dim = action_config.get("keyboard_dim_in", 4)

    action_presets = create_action_presets(num_frames, keyboard_dim)

    if action_preset not in action_presets:
        print(f"Warning: Preset '{action_preset}' not available for keyboard_dim={keyboard_dim}")
        print(f"Available presets: {list(action_presets.keys())}")
        action_preset = list(action_presets.keys())[0]
        print(f"Using: {action_preset}")

    actions = action_presets[action_preset]
    keyboard_cond = actions["keyboard"]
    mouse_cond = actions.get("mouse", torch.zeros(num_frames, 2))

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
        output_path=str(Path(OUTPUT_PATH) / f"output_{action_preset}.mp4"),
        save_video=True,
    )

if __name__ == "__main__":
    main()
