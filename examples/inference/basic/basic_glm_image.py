# SPDX-License-Identifier: Apache-2.0
"""
GLM-Image inference example using FastVideo.

This example demonstrates how to generate images using the GLM-Image model
through FastVideo's VideoGenerator API.

Usage:
    python examples/inference/basic/basic_glm_image.py
"""

import os

from PIL import Image

from fastvideo import VideoGenerator

OUTPUT_PATH = "image_output"


def main():
    # FastVideo will automatically use the optimal default arguments for GLM-Image.
    generator = VideoGenerator.from_pretrained(
        "zai-org/GLM-Image",
        num_gpus=1,
        trust_remote_code=True,
    )

    # Generate images - use save_video=False and return_frames=True
    # to get raw frames that we can save as images
    prompt = (
        "A beautiful landscape photography with rolling hills, "
        "a winding river, and a vibrant sunset in the background. "
        "Warm golden light, photorealistic style."
    )
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Generate first image
    result = generator.generate_video(
        prompt,
        output_path=OUTPUT_PATH,
        save_video=False,
        return_frames=True,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=7.5,  
    )
    
    # Save as PNG image (result is a list of frames, take the first/only one)
    if result and len(result) > 0:
        img = Image.fromarray(result[0])
        img.save(os.path.join(OUTPUT_PATH, "landscape.png"))
        print(f"Saved image to {OUTPUT_PATH}/landscape.png")


if __name__ == "__main__":
    main()
