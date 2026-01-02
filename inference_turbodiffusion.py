#!/usr/bin/env python3
"""
TurboDiffusion Inference Script for FastVideo

Minimal script for 1-4 step video generation using:
- TurboDiffusionPipeline with RCM scheduler
- SLA attention backend
- Auto-downloaded TurboDiffusion checkpoint weights

Usage:
    python inference_turbodiffusion.py --prompt "A curious raccoon..."
    python inference_turbodiffusion.py --prompts_file turbodiffusion_prompts.txt
"""

import os
import argparse

# Set SLA attention backend BEFORE fastvideo imports
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TurboDiffusion Video Generation")
    parser.add_argument("--prompt", type=str, default="A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest", help="Single text prompt")
    parser.add_argument("--prompts_file", type=str, default="turbodiffusion_prompts.txt",
                        help="Path to file with prompts (one per line)")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Steps (1-4)")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_path", type=str, default="outputs/", help="Output dir")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs")
    return parser.parse_args()


def load_prompts(args):
    """Load prompts from file or use single prompt."""
    if args.prompt:
        return [args.prompt]
    
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        return prompts
    
    raise ValueError("No prompt provided. Use --prompt or --prompts_file")


def main():
    args = parse_args()
    prompts = load_prompts(args)

    # Create video generator with TurboDiffusion pipeline
    # Now loads directly from pre-converted HF repo (no .pth loading needed)
    generator = VideoGenerator.from_pretrained(
        "loayrashid/TurboWan2.1-T2V-1.3B-Diffusers",  # Pre-converted weights
        num_gpus=args.num_gpus,
        override_pipeline_cls_name="TurboDiffusionPipeline",
    )

    # Generate video for each prompt
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating video {i+1}/{len(prompts)}: {prompt[:80]}...")
        
        # Note: guidance_scale=1.0 disables CFG - TurboDiffusion is distilled without CFG
        generator.generate_video(
            prompt,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            seed=args.seed,
            output_path=args.output_path,
            save_video=True,
            guidance_scale=1.0,  # TurboDiffusion doesn't use CFG
        )


if __name__ == "__main__":
    main()

