# SPDX-License-Identifier: Apache-2.0
"""
Basic inference script for HunyuanGameCraft video generation.

HunyuanGameCraft generates game-like videos with camera/action control.
It takes an optional image input and generates video with camera motion
based on simple action commands (forward, left, right, backward, rotations).

Usage:
    # Text-to-video with forward motion
    python basic_gamecraft.py --prompt "A medieval village with cobblestone streets" --action forward
    
    # Image-to-video with right motion
    python basic_gamecraft.py --image path/to/image.png --prompt "A forest path" --action right
    
    # Custom action speed
    python basic_gamecraft.py --prompt "A temple corridor" --action forward --action-speed 0.3

Available actions:
    - forward (w): Move camera forward
    - backward (s): Move camera backward  
    - left (a): Move camera left (strafe)
    - right (d): Move camera right (strafe)
    - left_rot: Rotate camera left (pan)
    - right_rot: Rotate camera right (pan)
    - up_rot: Rotate camera up (tilt)
    - down_rot: Rotate camera down (tilt)
"""
import argparse

import torch

from fastvideo import VideoGenerator
from fastvideo.models.camera import create_camera_trajectory

# Default prompts for demo
DEFAULT_PROMPTS = {
    "village": "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky.",
    "temple": "A majestic ancient temple stands under a clear blue sky, its grandeur highlighted by towering Doric columns and intricate architectural details.",
    "forest": "A lush green forest with tall trees, dappled sunlight filtering through the leaves, and a winding dirt path.",
    "beach": "A tropical beach with crystal clear turquoise water, white sand, and palm trees swaying in the breeze.",
}

# Action mapping (human-readable -> official action ID)
ACTION_MAP = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "left_rot": "left_rot",
    "right_rot": "right_rot",
    "up_rot": "up_rot",
    "down_rot": "down_rot",
}

OUTPUT_PATH = "video_samples_gamecraft"


def main():
    parser = argparse.ArgumentParser(
        description="HunyuanGameCraft video generation with camera/action control"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="tencent/HunyuanGameCraft",  # HuggingFace model path
        help="Path to GameCraft model (HuggingFace path or local directory)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPTS["village"],
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional input image path for image-to-video generation",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="forward",
        choices=list(ACTION_MAP.keys()),
        help="Camera motion action",
    )
    parser.add_argument(
        "--action-speed",
        type=float,
        default=0.2,
        help="Speed of camera motion (default: 0.2)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=33,
        help="Number of frames (33 or 65 recommended)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Video height in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width in pixels",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for inference",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=OUTPUT_PATH,
        help="Output directory for generated videos",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for output video",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offloading to save GPU memory",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for classifier-free guidance (e.g., 'blurry, low quality, distorted')",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HunyuanGameCraft Video Generation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Prompt: {args.prompt[:80]}...")
    print(f"Action: {args.action} (speed: {args.action_speed})")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Frames: {args.num_frames}")
    print(f"Steps: {args.num_inference_steps}")
    print(f"Seed: {args.seed}")
    print(f"Negative prompt: {args.negative_prompt or '(empty)'}")
    print(f"Output: {args.output_path}")
    print("=" * 60)

    # Initialize generator
    print("\nInitializing VideoGenerator for GameCraft...")
    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=args.num_gpus,
        use_fsdp_inference=True,
        dit_cpu_offload=args.cpu_offload,
        vae_cpu_offload=args.cpu_offload,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    # Create camera trajectory
    print(f"\nGenerating camera trajectory for '{args.action}' motion...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_states = create_camera_trajectory(
        action=args.action,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        action_speed=args.action_speed,
        device=device,
        dtype=torch.bfloat16,
    )
    print(f"Camera states shape: {camera_states.shape}")

    # Generate video
    print("\nGenerating video...")
    result = generator.generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image_path=args.image,
        camera_states=camera_states,
        output_path=args.output_path,
        save_video=True,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        fps=args.fps,
        seed=args.seed,
        guidance_scale=6.0,  # Official GameCraft uses CFG with guidance_scale=6.0
    )

    print(f"\nVideo saved to: {args.output_path}")
    print("Done!")

    # Optionally generate another video with different action
    print("\n" + "=" * 60)
    print("Generating second video with 'right' motion...")
    print("=" * 60)
    
    camera_states_right = create_camera_trajectory(
        action="right",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        action_speed=args.action_speed,
        device=device,
        dtype=torch.bfloat16,
    )
    
    result2 = generator.generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image_path=args.image,
        camera_states=camera_states_right,
        output_path=args.output_path,
        save_video=True,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        fps=args.fps,
        seed=args.seed + 1,  # Different seed for variety
        guidance_scale=6.0,
    )
    
    print(f"\nSecond video saved to: {args.output_path}")
    print("All done!")


if __name__ == "__main__":
    main()
