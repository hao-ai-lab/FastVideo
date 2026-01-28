#!/usr/bin/env python3
"""
Generate reference image using official diffusers GlmImagePipeline.

This script generates a golden reference image that will be committed to the repo
for SSIM regression testing.
"""

import os
import hashlib
import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image

# Test prompts (matching test_glm_image_similarity.py)
TEST_PROMPTS = [
    "A beautiful landscape photography with rolling hills, a winding river, and a vibrant sunset in the background. Warm golden light, photorealistic style.",
    "A majestic mountain landscape at golden hour with a calm lake reflecting snow-capped peaks",
]

# Fixed reference filenames matching the SSIM test.
REFERENCE_FILENAMES = {
    TEST_PROMPTS[0]: "landscape_ref.png",
    TEST_PROMPTS[1]: "mountain_ref.png",
}

# GLM-Image parameters (matching test_glm_image_similarity.py)
GLM_IMAGE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 1024,
}

# Determine device folder
device_name = torch.cuda.get_device_name()
if "A40" in device_name:
    device_folder = "A40_reference_videos"
elif "L40S" in device_name:
    device_folder = "L40S_reference_videos"
elif "H100" in device_name:
    device_folder = "H100_reference_videos"
else:
    print(f"Warning: Unsupported device {device_name}, using H100_reference_videos")
    device_folder = "H100_reference_videos"

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
reference_dir = os.path.join(base_dir, "fastvideo", "tests", "ssim", device_folder, "glm_image")
os.makedirs(reference_dir, exist_ok=True)

print(f"Loading GLM-Image pipeline from zai-org/GLM-Image...")
pipe = GlmImagePipeline.from_pretrained(
    "zai-org/GLM-Image",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

print(f"Generating reference images...")
for prompt in TEST_PROMPTS:
    image_filename = REFERENCE_FILENAMES[prompt]
    reference_path = os.path.join(reference_dir, image_filename)
    
    print(f"\nGenerating: {prompt[:50]}...")
    print(f"  Output: {reference_path}")
    
    generator = torch.Generator(device="cuda").manual_seed(GLM_IMAGE_PARAMS["seed"])
    
    result = pipe(
        prompt=prompt,
        height=GLM_IMAGE_PARAMS["height"],
        width=GLM_IMAGE_PARAMS["width"],
        num_inference_steps=GLM_IMAGE_PARAMS["num_inference_steps"],
        guidance_scale=GLM_IMAGE_PARAMS["guidance_scale"],
        generator=generator,
    )
    
    img = result.images[0]
    img.save(reference_path)
    print(f"Saved to {reference_path}")

print(f"\n✓ All reference images generated in {reference_dir}")
