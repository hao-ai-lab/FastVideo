# SPDX-License-Identifier: Apache-2.0
"""Run Flux2 Klein text-to-image generation through FastVideo.

User story:
    "I need a short local smoke for the Flux2 Klein checkpoint before wiring it
    into an image workflow. Use the model's distilled four-step defaults and
    write a single PNG so I can compare the output against the reference."
"""

from __future__ import annotations

import argparse
import os

from fastvideo import VideoGenerator


DEFAULT_PROMPT = "a brushed steel espresso machine on a marble counter, morning window light"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flux2 Klein text-to-image generation.")
    parser.add_argument(
        "--model-path",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HF id or local diffusers-format Flux2 Klein weights directory.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/flux2/flux2_klein.png",
        help="PNG output path or output directory.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text.")
    parser.add_argument("--seed", type=int, default=0, help="Generation seed.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--steps", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--backend",
        default=None,
        help="Set FASTVIDEO_ATTENTION_BACKEND, for example TORCH_SDPA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.backend

    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=args.num_gpus,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )
    try:
        generator.generate_video(
            prompt=args.prompt,
            output_path=args.output_path,
            save_video=True,
            height=args.height,
            width=args.width,
            num_frames=1,
            fps=1,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            seed=args.seed,
        )
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
