# SPDX-License-Identifier: Apache-2.0
"""
Basic inference script for HunyuanGameCraft video generation.

HunyuanGameCraft generates game-like videos with camera/action control.
It takes an optional image input and generates video with camera motion
based on simple action commands (forward, left, right, backward, rotations).

Available actions:
    - forward (w): Move camera forward
    - backward (s): Move camera backward  
    - left (a): Move camera left (strafe)
    - right (d): Move camera right (strafe)
    - left_rot: Rotate camera left (pan)
    - right_rot: Rotate camera right (pan)
    - up_rot: Rotate camera up (tilt)
    - down_rot: Rotate camera down (tilt)

T2V vs I2V:
    - Default: I2V (uses a default reference image). Set GAMECRAFT_I2V_IMAGE to a
      URL or path to use a different image.
    - T2V only (no reference image): run with GAMECRAFT_I2V_IMAGE= (empty).
"""
import os

import torch

from fastvideo import VideoGenerator
from fastvideo.models.camera import create_camera_trajectory

# Model configuration (use GAMECRAFT_MODEL_PATH for local weights)
MODEL_PATH = os.environ.get("GAMECRAFT_MODEL_PATH", "FastVideo/HunyuanGameCraft-Diffusers")

# Default prompts for demo
DEFAULT_PROMPTS = {
    "village": "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky.",
    "temple": "A majestic ancient temple stands under a clear blue sky, its grandeur highlighted by towering Doric columns and intricate architectural details.",
    "forest": "A lush green forest with tall trees, dappled sunlight filtering through the leaves, and a winding dirt path.",
    "beach": "A tropical beach with crystal clear turquoise water, white sand, and palm trees swaying in the breeze.",
}

# I2V: default reference image (URL). Can override with a local path.
DEFAULT_I2V_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/diffusers/astronaut.jpg"
)
DEFAULT_I2V_PROMPT = (
    "An astronaut hatching from an egg, on the surface of the moon, "
    "the darkness and depth of space realised in the background."
)

OUTPUT_PATH = "video_samples_gamecraft"


def main():
    # Initialize generator
    # FastVideo will automatically download weights from HuggingFace
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    # Video parameters
    height = 704
    width = 1280
    num_frames = 33
    action = "forward"
    action_speed = 0.2

    # Create camera trajectory (Plücker coordinates)
    camera_states = create_camera_trajectory(
        action=action,
        height=height,
        width=width,
        num_frames=num_frames,
        action_speed=action_speed,
        dtype=torch.bfloat16,
    )
    print(f"Camera states shape: {camera_states.shape}")

    # I2V vs T2V: unset GAMECRAFT_I2V_IMAGE -> I2V (default image). Set to "" -> T2V.
    env_image = os.environ.get("GAMECRAFT_I2V_IMAGE")
    if env_image is None:
        image_path = DEFAULT_I2V_IMAGE_URL  # default: I2V
    elif env_image.strip() == "":
        image_path = None  # T2V
    else:
        image_path = env_image.strip()  # I2V with given URL/path

    is_i2v = image_path is not None
    prompt = DEFAULT_I2V_PROMPT if is_i2v else DEFAULT_PROMPTS["temple"]
    print(f"Mode: {'I2V' if is_i2v else 'T2V'}, prompt: {prompt[:60]}...")

    gen_kw = dict(
        prompt=prompt,
        negative_prompt="",
        camera_states=camera_states,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=50,
        guidance_scale=6.0,
        seed=42,
        fps=24,
        output_path=OUTPUT_PATH,
        save_video=True,
    )
    if is_i2v:
        gen_kw["image_path"] = image_path
    generator.generate_video(**gen_kw)


if __name__ == "__main__":
    main()
