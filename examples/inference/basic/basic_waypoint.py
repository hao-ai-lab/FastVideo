# SPDX-License-Identifier: Apache-2.0
"""
Non-interactive Waypoint inference (e.g. RunPod / headless runs).

Generates a short video with fixed controls and saves it to an MP4.
Use this when you cannot provide interactive keyboard/mouse input
(e.g. SSH, Jupyter, or CI). View the result in Jupyter Lab file browser,
or download via SCP.

Usage (e.g. on RunPod):
  1. SSH:  ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
  2. Clone: git clone https://github.com/.../FastVideo.git && cd FastVideo
  3. Install: pip install -e .   (if "Cannot uninstall blinker" use:
            pip install -e . --ignore-installed blinker   or use a venv)
  4. Run:   python examples/inference/basic/basic_waypoint.py
  5. Video: video_samples_waypoint_runpod/waypoint_runpod.mp4

In Jupyter Lab (port 8888): run this in a notebook cell, then open
the output path in the file browser and play the video.

  python examples/inference/basic/basic_waypoint.py --num_steps 10

If the video looks blurry: use --video-quality 9 or 10 for sharper MP4 encoding,
--num_inference_steps 4 (default), and --height 368 --width 640 (multiples of 16).
Set WAYPOINT_DEBUG=1 to enable extensive tensor logging for blur diagnosis.
Use --debug-output fastvideo_debug.json to save structured stats (with WAYPOINT_DEBUG).
Use --output waypoint_7.mp4 (etc.) to keep multiple runs. --seed for repro.

Memory: The model can exceed 16GB VRAM (the official HF Space also hits this).
Use --low-memory to enable CPU offload and smaller buffers so it fits in 16GB.
"""

# Avoid PyTorch 2.10 + Triton "duplicate template name" on some RunPod images.
# Must be set before torch/fastvideo are imported. If you still see the error,
# run from shell: TORCH_COMPILE_DISABLE=1 python examples/.../basic_waypoint.py
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse

import torch

from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator

OUTPUT_DIR = "video_samples_waypoint_runpod"
OUTPUT_MP4 = "waypoint_runpod_1000.mp4"

# Owl-Control keycodes: W=forward, A=left, S=back, D=right, Space=jump
KEY_FORWARD = 17   # W


def main():
    parser = argparse.ArgumentParser(description="Waypoint non-interactive run")
    parser.add_argument(
        "--model",
        default="FastVideo/Waypoint-1-Small-Diffusers",
        help="HuggingFace repo with Waypoint model_index.json",
    )
    parser.add_argument(
        "--prompt",
        default="A first-person gameplay video exploring a stylized world.",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=60,
        help="Number of streaming steps. Each step produces --frames-per-step frames (default 1). At 60 fps: 60 steps = 1 s, 120 = 2 s, 7200 = 2 min.",
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=1,
        help="Frames generated per step (same action repeated). 1 = one frame per step; 8 = eight frames per step (fewer steps for same length).",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Directory to save the output video",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename (e.g. waypoint_fp32_test.mp4). Default: waypoint_runpod_70.mp4",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=368,
        help="Frame height (multiple of 16; use 368 not 360 to avoid VAE shape mismatch)",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Frame width (multiple of 16)"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=4, help="Denoising steps per frame"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable CPU offload and smaller buffers to fit in 16GB VRAM (slower).",
    )
    parser.add_argument(
        "--video-quality",
        "--video_quality",
        dest="video_quality",
        type=int,
        default=8,
        help="MP4 encoding quality 0-10 (higher = sharper, less blur). Default 8.",
    )
    parser.add_argument(
        "--debug-output",
        default=None,
        help="Save debug tensor stats to JSON (enables WAYPOINT_DEBUG). Compare with official.",
    )
    args = parser.parse_args()

    if args.debug_output:
        os.environ["WAYPOINT_DEBUG"] = "1"
        os.environ["WAYPOINT_DEBUG_FILE"] = os.path.abspath(args.debug_output)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Waypoint inference.")

    # Waypoint VAE decoder expects latent from 32x32; use 368 to avoid 32 vs 48 mismatch
    if args.height == 360:
        print("Warning: --height 360 can cause tensor shape mismatch; using 368.")
        args.height = 368

    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = args.output if args.output else OUTPUT_MP4
    output_path = os.path.join(args.output_dir, output_filename)

    # Low-memory: CPU offload and smaller buffer so we fit in 16GB (e.g. RunPod/HF)
    dit_offload = args.low_memory
    vae_offload = args.low_memory
    text_offload = args.low_memory
    pin_cpu = not args.low_memory
    total_frames = args.num_steps * args.frames_per_step
    num_frames_cap = max(32, total_frames + 8) if args.low_memory else max(120, total_frames + 8)

    print(f"Loading Waypoint from {args.model}...")
    if args.low_memory:
        print("Low-memory mode: CPU offload enabled (slower but fits 16GB).")
    generator = StreamingVideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=dit_offload,
        vae_cpu_offload=vae_offload,
        text_encoder_cpu_offload=text_offload,
        pin_cpu_memory=pin_cpu,
    )

    print("Resetting streaming session...")
    reset_kw = dict(
        prompt=args.prompt,
        num_frames=num_frames_cap,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        output_path=output_path,
        save_video=True,
        video_quality=args.video_quality,
    )
    if args.seed is not None:
        reset_kw["seed"] = args.seed
    generator.reset(**reset_kw)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Fixed control: hold "forward" (W). Shape (1, T, 256) / (1, T, 2) for T frames per step.
    keyboard_cond = torch.zeros((1, args.frames_per_step, 256), dtype=torch.float32)
    keyboard_cond[:, :, KEY_FORWARD] = 1.0
    mouse_cond = torch.zeros((1, args.frames_per_step, 2), dtype=torch.float32)

    print(f"Generating {total_frames} frames ({args.num_steps} steps x {args.frames_per_step} frames/step)...")
    for i in range(args.num_steps):
        _frames, _ = generator.step(keyboard_cond=keyboard_cond, mouse_cond=mouse_cond)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  step {i + 1}/{args.num_steps}")

    saved = generator.finalize()
    generator.shutdown()

    if saved:
        print(f"Done. Video saved to: {os.path.abspath(saved)}")
        print("  View in Jupyter Lab file browser, or download with scp.")
    else:
        print("No frames were saved.")


if __name__ == "__main__":
    main()
