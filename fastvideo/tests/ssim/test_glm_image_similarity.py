# SPDX-License-Identifier: Apache-2.0
"""
SSIM-based similarity tests for GLM-Image generation.

Tests GLM-Image text-to-image generation with various prompts and backends.
"""

import os

import pytest
import torch
import hashlib

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Device-specific reference folder
device_name = torch.cuda.get_device_name()
device_reference_folder_suffix = "_reference_videos"

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
elif "H100" in device_name:
    device_reference_folder = "H100" + device_reference_folder_suffix
else:
    logger.warning(f"Unsupported device for ssim tests: {device_name}")

# Test prompts
TEST_PROMPTS = [
    "A beautiful landscape photography with rolling hills, a winding river, and a vibrant sunset in the background. Warm golden light, photorealistic style.",
    "A majestic mountain landscape at golden hour with a calm lake reflecting snow-capped peaks",
]

# Fixed reference filenames (avoid hashing prompts)
REFERENCE_FILENAMES = {
    TEST_PROMPTS[0]: "landscape_ref.png"
}

# GLM-Image parameters
GLM_IMAGE_PARAMS = {
    "num_gpus": 1,
    "model_path": "zai-org/GLM-Image",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 1024,
}

# Minimum acceptable SSIM threshold
MIN_ACCEPTABLE_SSIM = 0.98


def _compute_image_ms_ssim(reference_path: str, generated_path: str) -> float:
    from PIL import Image
    import numpy as np

    img1 = Image.open(reference_path).convert("RGB")
    img2 = Image.open(generated_path).convert("RGB")
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)

    t1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).contiguous().float() / 255.0
    t2 = torch.from_numpy(np.array(img2)).permute(2, 0, 1).contiguous().float() / 255.0
    t1 = t1.unsqueeze(0)
    t2 = t2.unsqueeze(0)
    if torch.cuda.is_available():
        t1 = t1.cuda()
        t2 = t2.cuda()
    with torch.no_grad():
        return float(_ssim_torch(t1, t2).item())


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_1d = g.view(1, 1, 1, window_size)
    window_2d = (window_1d.transpose(-1, -2) @ window_1d).squeeze(0).squeeze(0)
    return window_2d


def _ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Compute SSIM for tensors in [0,1], shape [N,C,H,W].
    Returns per-sample SSIM averaged over channels.
    """
    assert x.shape == y.shape and x.dim() == 4
    device, dtype = x.device, x.dtype
    window = _gaussian_window(window_size, sigma, device, dtype).view(1, 1, window_size, window_size)
    padding = window_size // 2

    C = x.shape[1]
    window = window.expand(C, 1, window_size, window_size)

    mu_x = torch.nn.functional.conv2d(x, window, padding=padding, groups=C)
    mu_y = torch.nn.functional.conv2d(y, window, padding=padding, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = torch.nn.functional.conv2d(x * x, window, padding=padding, groups=C) - mu_x2
    sigma_y2 = torch.nn.functional.conv2d(y * y, window, padding=padding, groups=C) - mu_y2
    sigma_xy = torch.nn.functional.conv2d(x * y, window, padding=padding, groups=C) - mu_xy

    C1 = (0.01**2)
    C2 = (0.03**2)
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean(dim=[2, 3]).mean(dim=1)


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["TORCH_SDPA"])
def test_glm_image_similarity(prompt, ATTENTION_BACKEND):
    """
    Test GLM-Image generation similarity against reference images.
    
    Args:
        prompt: Text prompt for image generation
        ATTENTION_BACKEND: Attention backend to use (TORCH_SDPA for GLM-Image)
    """
    import os
    
    # Set attention backend
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND
    
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reference_dir = os.path.join(base_dir, device_reference_folder, "glm_image")
    generated_dir = os.path.join(base_dir, "generated_images")
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    
    # Generate filename from prompt
    image_filename = REFERENCE_FILENAMES.get(prompt)
    if image_filename is None:
        pytest.fail(f"No reference filename configured for prompt: {prompt}")
    reference_path = os.path.join(reference_dir, image_filename)
    generated_path = os.path.join(generated_dir, image_filename)
    
    # Initialize generator
    generator = VideoGenerator.from_pretrained(
        GLM_IMAGE_PARAMS["model_path"],
        num_gpus=GLM_IMAGE_PARAMS["num_gpus"],
        trust_remote_code=True,
    )
    
    try:
        # Generate image
        result = generator.generate_video(
            prompt=prompt,
            output_path=generated_dir,
            save_video=False,
            return_frames=True,
            height=GLM_IMAGE_PARAMS["height"],
            width=GLM_IMAGE_PARAMS["width"],
            num_inference_steps=GLM_IMAGE_PARAMS["num_inference_steps"],
            guidance_scale=GLM_IMAGE_PARAMS["guidance_scale"],
            seed=GLM_IMAGE_PARAMS["seed"],
        )
        
        # Save generated image
        if result and len(result) > 0:
            from PIL import Image
            img = Image.fromarray(result[0])
            img.save(generated_path)
            logger.info(f"Generated image saved to {generated_path}")
        else:
            pytest.fail("No image generated")
        
        # Compare with reference if it exists
        if os.path.exists(reference_path):
            mean_ssim = _compute_image_ms_ssim(reference_path, generated_path)
            
            # Assert SSIM threshold
            assert mean_ssim >= MIN_ACCEPTABLE_SSIM, (
                f"SSIM value {mean_ssim:.4f} is below threshold {MIN_ACCEPTABLE_SSIM} "
                f"for GLM-Image with backend {ATTENTION_BACKEND}"
            )
            logger.info(f"✓ SSIM test passed: {mean_ssim:.4f} >= {MIN_ACCEPTABLE_SSIM}")
        else:
            logger.warning(
                f"Reference image not found at {reference_path}. "
                f"Generated image saved to {generated_path}. "
                f"Please review and move to reference directory if acceptable."
            )
            pytest.skip(f"Reference image not found: {reference_path}")
    
    finally:
        generator.shutdown()
