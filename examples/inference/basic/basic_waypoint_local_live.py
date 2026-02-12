# SPDX-License-Identifier: Apache-2.0
"""
Waypoint-1-Small with a live window: press WASD / Space to move, see frames update.

Run this on a machine with an NVIDIA GPU (and CUDA). Each keypress runs one
generation step and shows the new frame in an OpenCV window. Not real-time FPS
(one step can take 1–3 seconds); for true 20–30 FPS you need a very fast GPU
and/or a different backend.

Usage:
    python examples/inference/basic/basic_waypoint_local_live.py

Controls:
    W / A / S / D  – forward / left / back / right
    Space          – jump
    Q or Esc       – quit (optionally saves the clip to an MP4)
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import sys

import cv2
import torch

from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator

# Owl-Control keycodes (same as RunPod example)
KEY_FORWARD = 17   # W
KEY_LEFT = 30      # A
KEY_BACK = 31      # S
KEY_RIGHT = 32     # D
KEY_JUMP = 57      # Space

WINDOW_NAME = "Waypoint-1 (WASD to move, Q to quit)"


def main():
    parser = argparse.ArgumentParser(description="Waypoint local live window")
    parser.add_argument(
        "--model",
        default="FastVideo/Waypoint-1-Small-Diffusers",
        help="HuggingFace model repo",
    )
    parser.add_argument(
        "--prompt",
        default="A first-person gameplay video exploring a stylized world.",
        help="Text prompt",
    )
    parser.add_argument(
        "--height", type=int, default=368, help="Frame height (multiple of 16)"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Frame width (multiple of 16)"
    )
    parser.add_argument(
        "--save_on_quit",
        action="store_true",
        help="Save generated frames to an MP4 when you press Q",
    )
    parser.add_argument(
        "--output_path",
        default="waypoint_local_live.mp4",
        help="Path for saved video when --save_on_quit",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required. Run this on a machine with an NVIDIA GPU.")
        sys.exit(1)

    print(f"Loading {args.model}...")
    generator = StreamingVideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
    )

    generator.reset(
        prompt=args.prompt,
        num_frames=120,
        height=args.height,
        width=args.width,
        num_inference_steps=4,
        output_path=args.output_path if args.save_on_quit else None,
        save_video=args.save_on_quit,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # Start with a black frame until first step
    current = None
    step_count = 0

    print("WASD + Space to move, Q or Esc to quit.")
    if args.save_on_quit:
        print("Video will be saved to", args.output_path, "when you quit.")

    while True:
        if current is not None:
            # OpenCV expects BGR
            show = cv2.cvtColor(current, cv2.COLOR_RGB2BGR)
            cv2.imshow(WINDOW_NAME, show)

        key = cv2.waitKey(100 if current is None else 50)
        if key in (-1, 255):
            continue
        key_char = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or key_char == "q":
            break

        # Build keyboard condition from key
        keyboard_cond = torch.zeros((1, 256), dtype=torch.float32)
        if key_char == "w":
            keyboard_cond[0, KEY_FORWARD] = 1.0
        elif key_char == "a":
            keyboard_cond[0, KEY_LEFT] = 1.0
        elif key_char == "s":
            keyboard_cond[0, KEY_BACK] = 1.0
        elif key_char == "d":
            keyboard_cond[0, KEY_RIGHT] = 1.0
        elif key == 32:  # Space
            keyboard_cond[0, KEY_JUMP] = 1.0
        else:
            continue

        mouse_cond = torch.zeros((1, 2), dtype=torch.float32)

        print(f"  step {step_count + 1}...", end=" ", flush=True)
        try:
            frames, _ = generator.step(
                keyboard_cond=keyboard_cond, mouse_cond=mouse_cond
            )
        except Exception as e:
            print("Error:", e)
            break
        if frames:
            current = frames[-1]
            step_count += 1
            print("ok")

    cv2.destroyAllWindows()

    if args.save_on_quit and step_count > 0:
        path = generator.finalize()
        if path:
            print("Saved to", path)
    else:
        generator.finalize()

    generator.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
