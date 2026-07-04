#!/usr/bin/env python3
"""06 - Real Wan2.1-T2V-1.3B Diffusers on the v2 runtime.

This is the first real-model path for v2: it resolves the exact Hugging Face
Diffusers repo, stamps the component checkpoints onto the Wan2.1 card, loads
the torch/CUDA component adapters, and runs generation through the same driven
denoise loop used by the CPU toy.

Run:
    python3 v2_examples/inference/06_real_wan21_diffusers.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2 import VideoGenerator

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def main() -> None:
    generator = VideoGenerator.from_pretrained(MODEL_ID, num_gpus=1)
    result = generator.generate_video(
        "A small robot walking through a neon-lit rainy street.",
        num_inference_steps=4,
        num_frames=17,
        height=240,
        width=416,
        guidance_scale=3.0,
        seed=1234,
        output_path="outputs/v2_real_wan21",
        output_video_name="wan21_t2v_1_3b_v2_smoke",
        save_video=True,
    )
    print(result.video_path)


if __name__ == "__main__":
    main()
