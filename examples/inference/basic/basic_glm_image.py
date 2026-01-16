# SPDX-License-Identifier: Apache-2.0
"""
GLM-Image inference example using FastVideo.

This example demonstrates how to generate images using the GLM-Image model
through FastVideo's pipeline infrastructure.

Usage:
    python examples/inference/basic/basic_glm_image.py
"""

import os

# Set environment variables for single-GPU distributed setup
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

from fastvideo.pipelines.basic.glm_image import GlmImagePipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

OUTPUT_PATH = "image_output"



def main():
    # Load the GLM-Image pipeline
    pipe = GlmImagePipeline.from_pretrained(
        "zai-org/GLM-Image",
        num_gpus=1,
        trust_remote_code=True,
    )

    # Create a batch with generation parameters
    prompt = (
        "A beautiful landscape photography with rolling hills, "
        "a winding river, and a vibrant sunset in the background. "
        "Warm golden light, photorealistic style."
    )
    
    batch = ForwardBatch(
        data_type="image",
        prompt=prompt,
        negative_prompt="",
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=7.5,
        guidance_rescale=0.7,
        do_classifier_free_guidance=True,
        num_frames=1,
        seed=42,
    )
    
    # Generate the image
    result = pipe.forward(batch, pipe.fastvideo_args)
    
    # Output is in result.output as a tensor [B, C, T, H, W]
    print(f"Generated image tensor shape: {result.output.shape}")
    
    # Save the output image
    import torch
    from torchvision.utils import save_image
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # result.output is [B, C, 1, H, W] for images
    image_tensor = result.output.squeeze(2) # [B, C, H, W]
    save_path = os.path.join(OUTPUT_PATH, "output.png")
    save_image(image_tensor, save_path)
    print(f"Image saved to {save_path}")


if __name__ == "__main__":
    main()
