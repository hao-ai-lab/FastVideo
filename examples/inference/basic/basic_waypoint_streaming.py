# SPDX-License-Identifier: Apache-2.0
"""
Basic streaming inference example for Waypoint-1-Small world model.

Waypoint is an interactive world model that generates video frames in real-time
based on text prompts and controller inputs (keyboard, mouse, scroll).

Usage:
    python examples/inference/basic/basic_waypoint_streaming.py

Prerequisites:
    1. Download Waypoint-1-Small weights from Hugging Face
    2. Set WAYPOINT_MODEL_PATH environment variable (or use default path)

This example demonstrates:
    - Initializing the Waypoint pipeline
    - Streaming generation with real-time control inputs
    - Interactive keyboard/mouse control handling
"""

import os
import sys
import asyncio
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import torch

# Default model path
WAYPOINT_MODEL_PATH = os.getenv(
    "WAYPOINT_MODEL_PATH",
    "official_weights/Waypoint-1-Small"
)

# Output settings
OUTPUT_PATH = "video_samples_waypoint"


@dataclass
class CtrlInput:
    """Controller input for Waypoint world model.
    
    Attributes:
        button: Set of pressed button IDs (0-255). Uses Owl-Control keycodes.
        mouse: Tuple of (x, y) mouse velocity as floats.
        scroll: Scroll wheel value (-1, 0, or 1).
    """
    button: Set[int] = None
    mouse: Tuple[float, float] = (0.0, 0.0)
    scroll: float = 0.0
    
    def __post_init__(self):
        if self.button is None:
            self.button = set()


# Common keyboard mappings (Owl-Control keycodes)
WASD_KEYCODES = {
    'w': 17,   # Forward
    'a': 30,   # Left  
    's': 31,   # Backward
    'd': 32,   # Right
    ' ': 57,   # Space (jump)
}


def get_keyboard_input() -> CtrlInput:
    """Get keyboard input from user.
    
    In a real application, you would use a library like pynput
    to capture actual keyboard/mouse events.
    """
    print("\nEnter control input (or 'q' to quit):")
    print("  WASD - movement, Space - jump")
    print("  m <x> <y> - mouse velocity")
    print("  s <-1/0/1> - scroll")
    
    try:
        user_input = input("> ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return None
    
    if user_input == 'q':
        return None
    
    ctrl = CtrlInput()
    
    # Parse WASD keys
    for char in user_input:
        if char in WASD_KEYCODES:
            ctrl.button.add(WASD_KEYCODES[char])
    
    # Parse mouse input (m x y)
    if user_input.startswith('m '):
        parts = user_input[2:].split()
        if len(parts) >= 2:
            try:
                ctrl.mouse = (float(parts[0]), float(parts[1]))
            except ValueError:
                pass
    
    # Parse scroll input (s value)
    if user_input.startswith('s '):
        try:
            ctrl.scroll = float(user_input[2:])
        except ValueError:
            pass
    
    return ctrl


async def main():
    """Main streaming generation loop."""
    # Check if model path exists
    if not os.path.exists(WAYPOINT_MODEL_PATH):
        print(f"Error: Waypoint model not found at {WAYPOINT_MODEL_PATH}")
        print("Please download from: https://huggingface.co/Overworld/Waypoint-1-Small")
        sys.exit(1)
    
    print("=" * 60)
    print("Waypoint-1-Small Streaming Inference")
    print("=" * 60)
    
    # Import here to avoid import errors before CUDA check
    try:
        from fastvideo.pipelines.basic.waypoint import WaypointPipeline
        from fastvideo.pipelines.basic.waypoint.waypoint_pipeline import CtrlInput as PipelineCtrlInput
    except ImportError as e:
        print(f"Error importing Waypoint pipeline: {e}")
        print("Please ensure FastVideo is installed correctly.")
        sys.exit(1)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is required for Waypoint inference")
        sys.exit(1)
    
    print(f"\nUsing model: {WAYPOINT_MODEL_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # For now, demonstrate manual inference since full pipeline loading
    # requires additional setup (VAE, text encoder, etc.)
    print("\n" + "=" * 60)
    print("Waypoint Streaming Demo")
    print("=" * 60)
    
    # Load just the transformer for demonstration
    from fastvideo.models.dits.waypoint_transformer import WaypointWorldModel
    from fastvideo.configs.models.dits.waypoint_transformer import WaypointArchConfig
    from safetensors.torch import load_file
    
    print("\nLoading Waypoint transformer...")
    
    config = WaypointArchConfig()
    model = WaypointWorldModel(config)
    
    weights_path = os.path.join(WAYPOINT_MODEL_PATH, "transformer", "diffusion_pytorch_model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    print("✓ Model loaded successfully!")
    
    # Demo parameters
    batch_size = 1
    n_frames = 1
    prompt_len = 32
    
    # Create mock prompt embedding (in real usage, this would come from UMT5)
    prompt_emb = torch.randn(batch_size, prompt_len, config.prompt_embedding_dim, device=device, dtype=dtype)
    prompt_pad_mask = torch.ones(batch_size, prompt_len, device=device, dtype=torch.bool)
    
    # Scheduler sigmas (from config)
    sigmas = [1.0, 0.8609585762023926, 0.729332447052002, 0.3205108940601349, 0.0]
    
    print("\nStarting interactive generation demo...")
    print("(This demo shows the denoising loop without VAE decoding)")
    
    frame_index = 0
    max_frames = 10
    
    while frame_index < max_frames:
        ctrl = get_keyboard_input()
        
        if ctrl is None:
            print("Exiting...")
            break
        
        print(f"\nGenerating frame {frame_index + 1}/{max_frames}...")
        print(f"  Buttons: {ctrl.button}")
        print(f"  Mouse: {ctrl.mouse}")
        print(f"  Scroll: {ctrl.scroll}")
        
        # Convert control input to tensors
        mouse = torch.tensor([[list(ctrl.mouse)]], device=device, dtype=dtype)
        button = torch.zeros(1, 1, config.n_buttons, device=device, dtype=dtype)
        for b in ctrl.button:
            if 0 <= b < config.n_buttons:
                button[0, 0, b] = 1.0
        scroll = torch.tensor([[[float(ctrl.scroll > 0) - float(ctrl.scroll < 0)]]], device=device, dtype=dtype)
        
        # Initialize latent noise
        x = torch.randn(batch_size, n_frames, config.channels, config.height, config.width, device=device, dtype=dtype)
        frame_ts = torch.tensor([[frame_index]], device=device, dtype=torch.long)
        
        # Denoise through sigma schedule
        with torch.no_grad():
            for i in range(len(sigmas) - 1):
                sigma_curr = sigmas[i]
                sigma_next = sigmas[i + 1]
                sigma = torch.tensor([[sigma_curr]], device=device, dtype=dtype)
                
                v_pred = model(
                    x=x,
                    sigma=sigma,
                    frame_timestamp=frame_ts,
                    prompt_emb=prompt_emb,
                    prompt_pad_mask=prompt_pad_mask,
                    mouse=mouse,
                    button=button,
                    scroll=scroll,
                    kv_cache=None,
                )
                
                # Euler step
                x = x + (sigma_next - sigma_curr) * v_pred
        
        # Log latent stats (would normally decode with VAE)
        latent_mean = x.float().mean().item()
        latent_std = x.float().std().item()
        print(f"  Latent stats: mean={latent_mean:.4f}, std={latent_std:.4f}")
        
        frame_index += 1
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Generated {frame_index} frames")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

