# SPDX-License-Identifier: Apache-2.0
"""SSIM-based regression tests for GLM-Image generation."""

import os

import pytest
import torch
from PIL import Image
import numpy as np

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Device-specific reference folder
device_name = torch.cuda.get_device_name()
if "A40" in device_name:
    device_reference_folder = "A40_reference_videos"
elif "L40S" in device_name:
    device_reference_folder = "L40S_reference_videos"
elif "H100" in device_name:
    device_reference_folder = "H100_reference_videos"
else:
    logger.warning(f"Unsupported device: {device_name}, using L40S references")
    device_reference_folder = "L40S_reference_videos"

# Test configuration
TEST_PROMPT = "A beautiful landscape photography with rolling hills, a winding river, and a vibrant sunset in the background. Warm golden light, photorealistic style."
REFERENCE_FILENAME = "landscape_ref.png"
MIN_ACCEPTABLE_SSIM = 0.98

GLM_IMAGE_PARAMS = {
    "model_path": "zai-org/GLM-Image",
    "num_gpus": 1,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 1024,
}


def _compute_ssim(reference_path: str, generated_path: str) -> float:
    """Compute SSIM between two images."""
    img1 = Image.open(reference_path).convert("RGB")
    img2 = Image.open(generated_path).convert("RGB")
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)

    t1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(np.array(img2)).permute(2, 0, 1).float() / 255.0
    t1, t2 = t1.unsqueeze(0), t2.unsqueeze(0)
    if torch.cuda.is_available():
        t1, t2 = t1.cuda(), t2.cuda()

    # Gaussian window for SSIM
    window_size, sigma = 11, 1.5
    coords = torch.arange(window_size, device=t1.device, dtype=t1.dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (g.view(-1, 1) @ g.view(1, -1)).view(1, 1, window_size, window_size)
    window = window.expand(3, 1, window_size, window_size)
    padding = window_size // 2

    mu_x = torch.nn.functional.conv2d(t1, window, padding=padding, groups=3)
    mu_y = torch.nn.functional.conv2d(t2, window, padding=padding, groups=3)
    sigma_x2 = torch.nn.functional.conv2d(t1 * t1, window, padding=padding, groups=3) - mu_x * mu_x
    sigma_y2 = torch.nn.functional.conv2d(t2 * t2, window, padding=padding, groups=3) - mu_y * mu_y
    sigma_xy = torch.nn.functional.conv2d(t1 * t2, window, padding=padding, groups=3) - mu_x * mu_y

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return float(ssim_map.mean().item())


def test_glm_image_similarity():
    """Test GLM-Image output consistency against reference image."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reference_path = os.path.join(base_dir, device_reference_folder, "glm_image", REFERENCE_FILENAME)
    generated_dir = os.path.join(base_dir, "generated_images")
    generated_path = os.path.join(generated_dir, REFERENCE_FILENAME)
    os.makedirs(generated_dir, exist_ok=True)

    generator = VideoGenerator.from_pretrained(
        GLM_IMAGE_PARAMS["model_path"],
        num_gpus=GLM_IMAGE_PARAMS["num_gpus"],
        trust_remote_code=True,
    )

    try:
        result = generator.generate_video(
            prompt=TEST_PROMPT,
            output_path=generated_dir,
            save_video=False,
            return_frames=True,
            height=GLM_IMAGE_PARAMS["height"],
            width=GLM_IMAGE_PARAMS["width"],
            num_inference_steps=GLM_IMAGE_PARAMS["num_inference_steps"],
            guidance_scale=GLM_IMAGE_PARAMS["guidance_scale"],
            seed=GLM_IMAGE_PARAMS["seed"],
        )

        assert result and len(result) > 0, "No image generated"
        Image.fromarray(result[0]).save(generated_path)
        logger.info(f"Generated image saved to {generated_path}")

        if not os.path.exists(reference_path):
            pytest.skip(f"Reference image not found: {reference_path}")

        ssim = _compute_ssim(reference_path, generated_path)
        assert ssim >= MIN_ACCEPTABLE_SSIM, \
            f"SSIM {ssim:.4f} below threshold {MIN_ACCEPTABLE_SSIM}"
        logger.info(f"SSIM test passed: {ssim:.4f} >= {MIN_ACCEPTABLE_SSIM}")

    finally:
        generator.shutdown()
