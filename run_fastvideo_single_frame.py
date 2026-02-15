# SPDX-License-Identifier: Apache-2.0
"""Generate a single Waypoint frame with FastVideo using the same prompt and seed
as test_official_model.py, so you can compare official_test_frame.png vs
waypoint_single_image.png.

Usage (local or RunPod):
  python run_fastvideo_single_frame.py
  python run_fastvideo_single_frame.py --seed 123

Output: waypoint_single_image.png in the current directory.
"""
import contextlib
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import torch
from PIL import Image
import numpy as np

from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator

# Same prompt as test_official_model.py for a fair comparison
DEFAULT_PROMPT = "first person view of a grassy field with blue sky"
DEFAULT_SEED = 42
OUTPUT_PNG = "waypoint_single_image.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Generate one FastVideo Waypoint frame for comparison with official model"
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt (default: same as official test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--model",
        default="FastVideo/Waypoint-1-Small-Diffusers",
        help="HuggingFace model repo",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help="Denoising steps per frame (default 4)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    print(f"Loading Waypoint from {args.model}...")
    generator = StreamingVideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
    )

    print("Resetting streaming session (1 frame)...")
    generator.reset(
        prompt=args.prompt,
        num_frames=8,
        height=368,
        width=640,
        num_inference_steps=args.num_inference_steps,
        save_video=False,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    keyboard_cond = torch.zeros((1, 1, 256), dtype=torch.float32)
    mouse_cond = torch.zeros((1, 1, 2), dtype=torch.float32)

    print("Generating 1 frame...")
    frames, _ = generator.step(keyboard_cond=keyboard_cond,
                               mouse_cond=mouse_cond)

    temp_mp4 = "temp_single_frame.mp4"
    generator.finalize(output_path=temp_mp4)
    generator.shutdown()
    if os.path.isfile(temp_mp4):
        with contextlib.suppress(OSError):
            os.remove(temp_mp4)

    if not frames:
        raise SystemExit("No frames returned from pipeline.")

    img = frames[0]
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[-1] == 3:
            Image.fromarray(img).save(OUTPUT_PNG)
        else:
            Image.fromarray(img).save(OUTPUT_PNG)
    else:
        raise SystemExit(f"Unexpected frame type: {type(img)}")

    print(f"Saved {OUTPUT_PNG}")
    print(
        "Compare with official_test_frame.png (same prompt + seed in test_official_model.py)."
    )


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
