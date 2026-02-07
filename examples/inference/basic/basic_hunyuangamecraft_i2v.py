# SPDX-License-Identifier: Apache-2.0
"""
Image-to-Video example for HunyuanGameCraft using FastVideo.

HunyuanGameCraft is designed for video continuation with reference frames.
This example shows how to generate videos from a reference image with
camera control using Plücker coordinates.

Usage:
    python basic_hunyuangamecraft_i2v.py \
        --image /path/to/reference.jpg \
        --prompt "A game scene with camera moving forward" \
        --output video_output.mp4

Note: This requires converted HunyuanGameCraft weights.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def custom_meshgrid(*args):
    """PyTorch meshgrid with ij indexing."""
    return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device):
    """
    Compute Plücker coordinates for camera rays.
    
    Args:
        K: Camera intrinsics [B, V, 4] - (fx, fy, cx, cy) normalized
        c2w: Camera-to-world matrices [B, V, 4, 4]
        H: Height
        W: Width
        device: Device
        
    Returns:
        Plücker coordinates [B, V, H, W, 6]
    """
    B, V = K.shape[:2]
    
    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5
    
    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B, V, 1
    
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)
    
    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)
    
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    
    rays_dxo = torch.cross(rays_o, rays_d)  # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, V, H, W, 6)  # B, V, H, W, 6
    return plucker


def generate_motion_segment(current_pose, motion_type, value, duration=33):
    """Generate camera motion for a segment."""
    import math
    positions = []
    rotations = []
    
    if motion_type in ['forward', 'backward']:
        yaw_rad = np.radians(current_pose['rotation'][1])
        pitch_rad = np.radians(current_pose['rotation'][0])
        
        forward_vec = np.array([
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad)
        ])
        
        direction = 1 if motion_type == 'forward' else -1
        total_move = forward_vec * value * direction
        step = total_move / duration
        
        for i in range(1, duration + 1):
            new_pos = current_pose['position'] + step * i
            positions.append(new_pos.copy())
            rotations.append(current_pose['rotation'].copy())
            
        current_pose['position'] = positions[-1]
        
    elif motion_type in ['left', 'right']:
        import math
        yaw_rad = np.radians(current_pose['rotation'][1])
        right_vec = np.array([math.cos(yaw_rad), 0, -math.sin(yaw_rad)])
        
        direction = -1 if motion_type == 'right' else 1
        total_move = right_vec * value * direction
        step = total_move / duration
        
        for i in range(1, duration + 1):
            new_pos = current_pose['position'] + step * i
            positions.append(new_pos.copy())
            rotations.append(current_pose['rotation'].copy())
            
        current_pose['position'] = positions[-1]
    else:
        # Stationary - no movement
        for i in range(duration):
            positions.append(current_pose['position'].copy())
            rotations.append(current_pose['rotation'].copy())
    
    return positions, rotations, current_pose


def euler_to_quaternion(angles):
    """Convert euler angles to quaternion."""
    import math
    pitch, yaw, roll = np.radians(angles)
    
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    
    return [qw, qx, qy, qz]


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])


def create_camera_states(
    num_frames: int,
    height: int,
    width: int,
    trajectory_type: str = "forward",
    action_speed: float = 0.2,
) -> torch.Tensor:
    """
    Create proper Plücker coordinate camera states matching official implementation.
    
    Args:
        num_frames: Number of frames
        height: Video height  
        width: Video width
        trajectory_type: Type of camera movement
        action_speed: Speed of movement
    
    Returns:
        Camera states tensor of shape [1, num_frames, 6, height, width]
    """
    ACTION_DICT = {
        "forward": "forward", 
        "backward": "backward", 
        "left": "left", 
        "right": "right",
        "stationary": "stationary",
        "up": "forward",  # Map up/down to forward for now
        "down": "backward",
    }
    
    motion_type = ACTION_DICT.get(trajectory_type, "forward")
    
    # Generate motion poses
    current_pose = {
        'position': np.array([0.0, 0.0, 0.0]),
        'rotation': np.array([0.0, 0.0, 0.0])
    }
    
    positions, rotations, _ = generate_motion_segment(
        current_pose, motion_type, action_speed, num_frames
    )
    
    # Camera intrinsics (normalized)
    intrinsic = [0.50505, 0.8979, 0.5, 0.5]  # fx, fy, cx, cy
    fx_norm = intrinsic[0] * width
    fy_norm = intrinsic[1] * height
    cx_norm = intrinsic[2] * width
    cy_norm = intrinsic[3] * height
    
    # Build c2w matrices from poses
    c2w_list = []
    
    # First frame is identity
    c2w_first = np.eye(4, dtype=np.float32)
    c2w_list.append(c2w_first)
    
    # Remaining frames based on motion
    for pos, rot in zip(positions, rotations):
        quat = euler_to_quaternion(rot)
        R = quaternion_to_rotation_matrix(quat)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos * 2  # Scale position like official code
        c2w_list.append(c2w)
    
    # Only take num_frames
    c2w_list = c2w_list[:num_frames]
    
    # Pad if needed
    while len(c2w_list) < num_frames:
        c2w_list.append(c2w_list[-1])
    
    c2w = torch.tensor(np.array(c2w_list, dtype=np.float32))[None]  # [1, F, 4, 4]
    
    # Camera intrinsics for all frames
    K = torch.tensor([[[fx_norm, fy_norm, cx_norm, cy_norm]]], dtype=torch.float32)
    K = K.expand(1, num_frames, 4)  # [1, F, 4]
    
    # Compute Plücker embeddings
    plucker = ray_condition(K, c2w, height, width, device='cpu')  # [1, F, H, W, 6]
    plucker = plucker[0].permute(0, 3, 1, 2)  # [F, 6, H, W]
    
    return plucker.unsqueeze(0)  # [1, F, 6, H, W]


def create_simple_camera_states(
    num_frames: int,
    height: int,
    width: int,
    trajectory_type: str = "forward",
) -> torch.Tensor:
    """
    Wrapper for backwards compatibility.
    """
    return create_camera_states(num_frames, height, width, trajectory_type, action_speed=0.2)


def load_and_preprocess_image(
    image_path: str,
    target_height: int,
    target_width: int,
) -> Image.Image:
    """Load and resize image to target dimensions."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image


def create_sample_image(height: int, width: int) -> Image.Image:
    """Create a sample gradient image for testing when no image is provided."""
    # Create a simple gradient image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a sky-to-ground gradient (blue to green)
    for y in range(height):
        t = y / height
        # Sky (blue) to ground (green/brown)
        r = int(100 * t + 50 * (1 - t))
        g = int(150 * t + 100 * (1 - t))
        b = int(50 * t + 200 * (1 - t))
        img_array[y, :, 0] = r
        img_array[y, :, 1] = g
        img_array[y, :, 2] = b
    
    return Image.fromarray(img_array)


# Default settings
DEFAULT_MODEL_PATH = "converted_weights/HunyuanGameCraft"
DEFAULT_PROMPT = (
    "A first-person view moving through a forest environment. "
    "Trees and foliage visible on both sides. "
    "Natural lighting with shadows."
)
DEFAULT_OUTPUT_PATH = "video_samples_hunyuangamecraft_i2v"


def main():
    parser = argparse.ArgumentParser(
        description="HunyuanGameCraft Image-to-Video generation with FastVideo"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to converted HunyuanGameCraft weights"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to reference image (if not provided, uses a sample gradient)"
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
        help="Output directory"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=37,
        help="Number of frames (default: 37; gives 10 latent frames via (37-1)//4+1=10, matching official's 10 latent frames)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
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
        help="Classifier-free guidance scale (official default is 6.0 with flow_shift=5.0)"
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
    print("HunyuanGameCraft Image-to-Video Generation")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Reference image: {args.image or 'Using sample gradient'}")
    print(f"Prompt: {args.prompt[:80]}...")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Frames: {args.num_frames}")
    print(f"Camera trajectory: {args.trajectory}")
    print(f"Output: {args.output_path}")
    print("=" * 60)
    
    # Load or create reference image
    if args.image and os.path.exists(args.image):
        print(f"\nLoading reference image from {args.image}...")
        ref_image = load_and_preprocess_image(args.image, args.height, args.width)
    else:
        print("\nNo reference image provided, creating sample gradient...")
        ref_image = create_sample_image(args.height, args.width)
    
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
    
    # Create camera trajectory.
    # The official implementation generates 33 camera poses (one per generated
    # frame, excluding the reference frame).  The CameraNet handles the
    # temporal mapping from 33 camera frames to latent frame tokens internally.
    camera_num_frames = 33  # Always 33 camera poses for single-image I2V
    print(f"\nCreating camera trajectory ({args.trajectory}, {camera_num_frames} camera frames)...")
    camera_states = create_simple_camera_states(
        num_frames=camera_num_frames,
        height=args.height,
        width=args.width,
        trajectory_type=args.trajectory,
    )
    # create_simple_camera_states already returns [1, F, 6, H, W]
    # No additional unsqueeze needed
    
    print(f"Camera states shape: {camera_states.shape}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Generate video
    print("\nGenerating video...")
    start_time = time.time()
    
    result = generator.generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        pil_image=ref_image,  # Pass reference image (field name is pil_image in SamplingParam)
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
