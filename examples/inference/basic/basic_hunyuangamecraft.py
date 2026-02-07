# SPDX-License-Identifier: Apache-2.0
"""
Basic example for HunyuanGameCraft video generation using FastVideo.

HunyuanGameCraft is a camera-controllable video generation model that uses
Plücker coordinate representation (6D ray format) for camera pose conditioning.

This example demonstrates:
1. Text-to-video generation with camera control
2. Using CameraNet for processing camera states
3. Integration with the FastVideo pipeline

Usage:
    python basic_hunyuangamecraft.py --prompt "A game scene" --output video_output.mp4

Note: This requires converted HunyuanGameCraft weights. If using official weights,
run the conversion script first:
    python scripts/checkpoint_conversion/convert_gamecraft_to_fastvideo.py \
        --source /path/to/official_weights/gamecraft_model_states.pt \
        --output converted_weights/HunyuanGameCraft
"""

import argparse
import time

import numpy as np
import torch

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def create_camera_trajectory(
    num_frames: int,
    height: int,
    width: int,
    trajectory_type: str = "forward",
) -> torch.Tensor:
    """
    Create camera trajectory using Plücker coordinates.
    
    Plücker coordinates represent a 3D line (ray) using 6 numbers:
    - First 3: Direction vector (d)
    - Last 3: Moment vector (m = p × d, where p is a point on the line)
    
    Args:
        num_frames: Number of frames in the video
        height: Height of the video in pixels
        width: Width of the video in pixels
        trajectory_type: Type of camera trajectory
            - "forward": Camera moves forward
            - "backward": Camera moves backward  
            - "left": Camera moves left
            - "right": Camera moves right
            - "stationary": Camera stays still
            - "orbit": Camera orbits around center
    
    Returns:
        Camera states tensor of shape [num_frames, 6, height, width]
    """
    # Create a grid of pixel coordinates
    y_coords = torch.linspace(-1, 1, height)
    x_coords = torch.linspace(-1, 1, width)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    camera_states = torch.zeros(num_frames, 6, height, width)
    
    for frame_idx in range(num_frames):
        t = frame_idx / max(num_frames - 1, 1)  # Normalized time [0, 1]
        
        # Define camera movement based on trajectory type
        if trajectory_type == "forward":
            # Camera moves forward (positive z direction)
            direction = torch.tensor([0.0, 0.0, 1.0])
            position = torch.tensor([0.0, 0.0, -3.0 + t * 2.0])
        elif trajectory_type == "backward":
            # Camera moves backward (negative z direction)
            direction = torch.tensor([0.0, 0.0, -1.0])
            position = torch.tensor([0.0, 0.0, -1.0 - t * 2.0])
        elif trajectory_type == "left":
            # Camera moves left (negative x direction)
            direction = torch.tensor([-1.0, 0.0, 0.0])
            position = torch.tensor([1.0 - t * 2.0, 0.0, -2.0])
        elif trajectory_type == "right":
            # Camera moves right (positive x direction)
            direction = torch.tensor([1.0, 0.0, 0.0])
            position = torch.tensor([-1.0 + t * 2.0, 0.0, -2.0])
        elif trajectory_type == "orbit":
            # Camera orbits around center
            angle = t * 2 * np.pi  # Full rotation
            radius = 3.0
            position = torch.tensor([
                radius * np.sin(angle),
                0.0,
                -radius * np.cos(angle)
            ])
            # Direction points toward center
            direction = -position / position.norm()
        else:  # stationary
            direction = torch.tensor([0.0, 0.0, 1.0])
            position = torch.tensor([0.0, 0.0, -3.0])
        
        # Normalize direction
        direction = direction / direction.norm()
        
        # Compute Plücker coordinates for each pixel
        # For each pixel, we create a ray from the camera through that pixel
        for y in range(height):
            for x in range(width):
                # Pixel direction in camera space (normalized device coordinates)
                pixel_dir = torch.tensor([
                    grid_x[y, x].item(),
                    grid_y[y, x].item(),
                    1.0
                ])
                pixel_dir = pixel_dir / pixel_dir.norm()
                
                # Transform to world space (simplified - assuming identity rotation)
                world_dir = pixel_dir
                
                # Moment vector: m = p × d
                moment = torch.cross(position, world_dir)
                
                # Store Plücker coordinates: [d, m]
                camera_states[frame_idx, 0:3, y, x] = world_dir
                camera_states[frame_idx, 3:6, y, x] = moment
    
    return camera_states


def create_simple_camera_states(
    num_frames: int,
    height: int,
    width: int,
    trajectory_type: str = "forward",
) -> torch.Tensor:
    """
    Create simplified camera states for testing.
    
    This creates camera states using a simpler approach that's more efficient
    for testing purposes. For production use, you may want to use more
    sophisticated camera trajectory generation.
    
    Args:
        num_frames: Number of frames
        height: Video height
        width: Video width
        trajectory_type: Type of camera movement
    
    Returns:
        Camera states tensor of shape [num_frames, 6, height, width]
    """
    camera_states = torch.zeros(num_frames, 6, height, width)
    
    # Movement vectors for different trajectories
    movement_configs = {
        "forward": (0.0, 0.0, 0.1),    # +z direction
        "backward": (0.0, 0.0, -0.1),  # -z direction
        "left": (-0.1, 0.0, 0.0),      # -x direction
        "right": (0.1, 0.0, 0.0),      # +x direction
        "up": (0.0, 0.1, 0.0),         # +y direction
        "down": (0.0, -0.1, 0.0),      # -y direction
        "stationary": (0.0, 0.0, 0.0), # no movement
    }
    
    move = movement_configs.get(trajectory_type, (0.0, 0.0, 0.0))
    
    for frame_idx in range(num_frames):
        t = frame_idx / max(num_frames - 1, 1)
        
        # Direction (first 3 channels) - forward direction with slight variation
        camera_states[frame_idx, 0, :, :] = move[0] * t  # x component
        camera_states[frame_idx, 1, :, :] = move[1] * t  # y component
        camera_states[frame_idx, 2, :, :] = 1.0 + move[2] * t  # z component
        
        # Moment (last 3 channels) - simplified
        camera_states[frame_idx, 3, :, :] = 0.0
        camera_states[frame_idx, 4, :, :] = 0.0
        camera_states[frame_idx, 5, :, :] = 0.0
    
    return camera_states


# Default settings
DEFAULT_MODEL_PATH = "converted_weights/HunyuanGameCraft"
DEFAULT_PROMPT = (
    "A first-person view of walking through a dense forest. "
    "Sunlight filters through the canopy, creating dappled shadows on the ground. "
    "The camera moves forward steadily."
)
DEFAULT_OUTPUT_PATH = "video_samples_hunyuangamecraft"


def main():
    parser = argparse.ArgumentParser(
        description="HunyuanGameCraft video generation with FastVideo"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to converted HunyuanGameCraft weights"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output video path"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=33,
        help="Number of frames (default: 33, ~1.3s at 24fps)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Video height in pixels"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width in pixels"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="forward",
        choices=["forward", "backward", "left", "right", "up", "down", "stationary"],
        help="Camera trajectory type"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offloading for memory-constrained setups"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HunyuanGameCraft Video Generation")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Prompt: {args.prompt[:100]}...")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Frames: {args.num_frames}")
    print(f"Camera trajectory: {args.trajectory}")
    print(f"Output: {args.output_path}")
    print("=" * 60)
    
    # Initialize generator
    print("\nInitializing VideoGenerator...")
    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=args.num_gpus,
        use_fsdp_inference=args.num_gpus > 1,
        dit_cpu_offload=args.cpu_offload,
        vae_cpu_offload=args.cpu_offload,
        text_encoder_cpu_offload=args.cpu_offload,
        pin_cpu_memory=True,
    )
    
    # Create camera trajectory
    print(f"\nCreating camera trajectory ({args.trajectory})...")
    camera_states = create_simple_camera_states(
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        trajectory_type=args.trajectory,
    )
    # Add batch dimension
    camera_states = camera_states.unsqueeze(0)  # [1, F, 6, H, W]
    
    print(f"Camera states shape: {camera_states.shape}")
    
    # Generate video
    print("\nGenerating video...")
    start_time = time.time()
    
    result = generator.generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        camera_states=camera_states,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_path=args.output_path,
        save_video=True,
    )
    
    elapsed = time.time() - start_time
    
    print(f"\nVideo generated successfully!")
    print(f"Saved to: {args.output_path}")
    print(f"Generation time: {elapsed:.2f}s")
    print(f"Frames per second: {args.num_frames / elapsed:.2f}")
    
    return result


if __name__ == "__main__":
    main()
