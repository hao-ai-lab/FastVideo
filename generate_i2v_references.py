#!/usr/bin/env python3
"""
Generate reference videos for TurboDiffusion I2V SSIM tests.

Run this script to create reference videos that will be used for SSIM comparison.
The generated videos should be copied to the appropriate reference folder:
  fastvideo/tests/ssim/{device}_reference_videos/TurboWan2.2-I2V-A14B-Diffusers/SLA_ATTN/
"""

import os

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"

from fastvideo import VideoGenerator

# Test parameters matching test_turbodiffusion_similarity.py
MODEL_PATH = "loayrashid/TurboWan2.2-I2V-A14B-Diffusers"
OUTPUT_DIR = "reference_videos_i2v"

PROMPTS = [
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
]

IMAGE_PATHS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating I2V reference videos to: {OUTPUT_DIR}")
    print(f"Model: {MODEL_PATH}")
    
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=2,
        sp_size=2,
        tp_size=1,
        override_pipeline_cls_name="TurboDiffusionI2VPipeline",
    )
    
    for prompt, image_path in zip(PROMPTS, IMAGE_PATHS):
        print(f"\nGenerating: {prompt[:60]}...")
        
        generator.generate_video(
            prompt,
            image_path=image_path,
            output_path=OUTPUT_DIR,
            height=720,
            width=1280,
            num_frames=81,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=42,
            fps=24,
        )
    
    print(f"\n✓ Done! Copy videos from {OUTPUT_DIR}/ to:")
    print("  fastvideo/tests/ssim/{L40S or A40}_reference_videos/TurboWan2.2-I2V-A14B-Diffusers/SLA_ATTN/")


if __name__ == "__main__":
    main()
