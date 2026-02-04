# SPDX-License-Identifier: Apache-2.0
"""
Basic streaming inference example for Waypoint-1-Small world model.

This mirrors other FastVideo streaming examples and uses
`StreamingVideoGenerator` so weights can be auto-downloaded from a HF repo
that contains a FastVideo-compatible `model_index.json`.

Usage:
    python examples/inference/basic/basic_waypoint_streaming.py
"""

import torch

from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
from fastvideo.pipelines.basic.waypoint.waypoint_pipeline import CtrlInput

OUTPUT_PATH = "video_samples_waypoint"


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


def main():
    """Main streaming generation loop."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Waypoint inference.")

    generator = StreamingVideoGenerator.from_pretrained(
        "FastVideo/Waypoint-1-Small-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
    )

    # Initialize streaming session.
    generator.reset(
        prompt="A first-person gameplay video exploring a stylized world.",
        num_frames=120,
        height=360,
        width=640,
        num_inference_steps=4,
        output_path=OUTPUT_PATH,
        save_video=True,
    )

    frame_index = 0
    max_steps = 50
    
    while frame_index < max_steps:
        ctrl = get_keyboard_input()
        
        if ctrl is None:
            print("Exiting...")
            break

        # Build 1-frame actions for StreamingVideoGenerator.step().
        keyboard_cond = torch.zeros((1, 256), dtype=torch.float32)
        for b in ctrl.button:
            if 0 <= b < 256:
                keyboard_cond[0, b] = 1.0
        mouse_cond = torch.tensor([list(ctrl.mouse)], dtype=torch.float32)

        _frames, _future = generator.step(keyboard_cond=keyboard_cond,
                                          mouse_cond=mouse_cond)
        frame_index += 1
    
    generator.finalize()
    generator.shutdown()


if __name__ == "__main__":
    main()

