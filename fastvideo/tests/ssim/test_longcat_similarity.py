# SPDX-License-Identifier: Apache-2.0
"""
SSIM-based similarity tests for LongCat video generation.

Tests three LongCat modes:
- T2V (Text-to-Video): 480p video from text prompt
- I2V (Image-to-Video): 480p video from image + text prompt  
- VC (Video Continuation): 480p video continuation from input video + text prompt

Sampling parameters are derived from:
- examples/inference/basic/basic_longcat_t2v.py
- examples/inference/basic/basic_longcat_i2v.py
- examples/inference/basic/basic_longcat_vc.py

Note: num_inference_steps is reduced for CI speed (4 steps vs 50 in examples).
"""
import os

import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.configs.sample.base import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    build_reference_folder_path,
    get_cuda_device_name,
    resolve_device_reference_folder,
    select_ssim_params,
)
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results

logger = init_logger(__name__)

REQUIRED_GPUS = 1

# Device-specific reference folder
device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H100", "H100"),
        ("H200", "H200"),
    ),
    device_name=get_cuda_device_name(),
    logger=logger,
)

# Common negative prompt from example scripts
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)

# =============================================================================
# LongCat T2V Parameters (from basic_longcat_t2v.py)
# =============================================================================
LONGCAT_T2V_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/LongCat-Video-T2V-Diffusers",
    "height": 480,
    "width": 480,
    "num_frames": 43,
    "num_inference_steps": 4,  # Reduced from 50 for CI speed
    "guidance_scale": 4.0,
    "fps": 15,
    "seed": 42,
    "negative_prompt": NEGATIVE_PROMPT,
}
_LONGCAT_T2V_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    LONGCAT_T2V_PARAMS["model_path"]
)
LONGCAT_T2V_FULL_QUALITY_PARAMS = {
    "num_gpus": LONGCAT_T2V_PARAMS["num_gpus"],
    "model_path": LONGCAT_T2V_PARAMS["model_path"],
    "height": _LONGCAT_T2V_FULL_QUALITY_DEFAULTS.height,
    "width": _LONGCAT_T2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": LONGCAT_T2V_PARAMS["num_frames"],  # default num_frames: 125
    "num_inference_steps": _LONGCAT_T2V_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _LONGCAT_T2V_FULL_QUALITY_DEFAULTS.guidance_scale,
    "fps": _LONGCAT_T2V_FULL_QUALITY_DEFAULTS.fps,
    "seed": _LONGCAT_T2V_FULL_QUALITY_DEFAULTS.seed,
    "negative_prompt": _LONGCAT_T2V_FULL_QUALITY_DEFAULTS.negative_prompt,
}

# =============================================================================
# LongCat I2V Parameters (from basic_longcat_i2v.py)
# =============================================================================
LONGCAT_I2V_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/LongCat-Video-I2V-Diffusers",
    "height": 480,
    "width": 480,  # Square for I2V
    "num_frames": 43,
    "num_inference_steps": 4,  # Reduced from 50 for CI speed
    "guidance_scale": 4.0,
    "fps": 15,
    "seed": 42,
    "negative_prompt": NEGATIVE_PROMPT,
}
_LONGCAT_I2V_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    LONGCAT_I2V_PARAMS["model_path"]
)
LONGCAT_I2V_FULL_QUALITY_PARAMS = {
    "num_gpus": LONGCAT_I2V_PARAMS["num_gpus"],
    "model_path": LONGCAT_I2V_PARAMS["model_path"],
    "height": _LONGCAT_I2V_FULL_QUALITY_DEFAULTS.height,
    "width": _LONGCAT_I2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": LONGCAT_I2V_PARAMS["num_frames"],  # default num_frames: 125
    "num_inference_steps": _LONGCAT_I2V_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _LONGCAT_I2V_FULL_QUALITY_DEFAULTS.guidance_scale,
    "fps": _LONGCAT_I2V_FULL_QUALITY_DEFAULTS.fps,
    "seed": _LONGCAT_I2V_FULL_QUALITY_DEFAULTS.seed,
    "negative_prompt": _LONGCAT_I2V_FULL_QUALITY_DEFAULTS.negative_prompt,
}

# =============================================================================
# LongCat VC Parameters (from basic_longcat_vc.py)
# =============================================================================
LONGCAT_VC_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/LongCat-Video-VC-Diffusers",
    "height": 480,
    "width": 480,
    "num_frames": 43,
    "num_inference_steps": 4,  # Reduced from 50 for CI speed
    "guidance_scale": 4.0,
    "fps": 15,
    "seed": 42,
    "num_cond_frames": 13,
    "negative_prompt": NEGATIVE_PROMPT,
}
_LONGCAT_VC_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    LONGCAT_VC_PARAMS["model_path"]
)
LONGCAT_VC_FULL_QUALITY_PARAMS = {
    "num_gpus": LONGCAT_VC_PARAMS["num_gpus"],
    "model_path": LONGCAT_VC_PARAMS["model_path"],
    "height": _LONGCAT_VC_FULL_QUALITY_DEFAULTS.height,
    "width": _LONGCAT_VC_FULL_QUALITY_DEFAULTS.width,
    "num_frames": LONGCAT_VC_PARAMS["num_frames"],  # default num_frames: 125
    "num_inference_steps": _LONGCAT_VC_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _LONGCAT_VC_FULL_QUALITY_DEFAULTS.guidance_scale,
    "fps": _LONGCAT_VC_FULL_QUALITY_DEFAULTS.fps,
    "seed": _LONGCAT_VC_FULL_QUALITY_DEFAULTS.seed,
    "num_cond_frames": LONGCAT_VC_PARAMS["num_cond_frames"],
    "negative_prompt": _LONGCAT_VC_FULL_QUALITY_DEFAULTS.negative_prompt,
}

# Test prompts
T2V_TEST_PROMPTS = [
    "In a realistic photography style, a white boy around seven or eight years old "
    "sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. "
    "He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a "
    "medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, "
    "who eagerly licks it with its tongue. The sun is shining brightly, and the background "
    "features a green lawn and several tall trees, creating a warm and loving scene.",
]

I2V_TEST_PROMPTS = [
    "A woman sits at a wooden table by the window in a cozy café. She reaches out "
    "with her right hand, picks up the white coffee cup from the saucer, and gently "
    "brings it to her lips to take a sip. After drinking, she places the cup back on "
    "the table and looks out the window, enjoying the peaceful atmosphere.",
]

I2V_IMAGE_PATHS = [
    "assets/girl.png",
]

VC_TEST_PROMPTS = [
    "A person rides a motorcycle along a long, straight road that stretches between "
    "a body of water and a forested hillside. The rider steadily accelerates, keeping "
    "the motorcycle centered between the guardrails, while the scenery passes by on "
    "both sides. The video captures the journey from the rider's perspective, emphasizing "
    "the sense of motion and adventure.",
]

VC_VIDEO_PATHS = [
    "assets/motorcycle.mp4",
]


def _resolve_asset_path(asset_path: str) -> str:
    """Resolve asset path relative to FastVideo root."""
    # Check if absolute or already exists
    if os.path.isabs(asset_path) or os.path.exists(asset_path):
        return asset_path
    # Try relative to workspace root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    return os.path.join(repo_root, asset_path)


@pytest.mark.parametrize("prompt", T2V_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
def test_longcat_t2v_similarity(prompt: str, ATTENTION_BACKEND: str):
    """
    Test LongCat T2V inference and compare output to reference videos using SSIM.
    
    Parameters derived from examples/inference/basic/basic_longcat_t2v.py
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    params = select_ssim_params(LONGCAT_T2V_PARAMS, LONGCAT_T2V_FULL_QUALITY_PARAMS)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "LongCat-Video-T2V"
    
    output_dir = build_generated_output_dir(
        script_dir,
        model_id,
        ATTENTION_BACKEND,
    )
    output_video_name = f"{prompt[:100].strip()}.mp4"
    os.makedirs(output_dir, exist_ok=True)

    init_kwargs = {
        "num_gpus": params["num_gpus"],
        "use_fsdp_inference": True,
        "dit_cpu_offload": True,
        "vae_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "enable_bsa": False,
    }

    generation_kwargs = {
        "output_path": output_dir,
        "height": params["height"],
        "width": params["width"],
        "num_frames": params["num_frames"],
        "num_inference_steps": params["num_inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "fps": params["fps"],
        "seed": params["seed"],
        "negative_prompt": params["negative_prompt"],
    }

    generator = VideoGenerator.from_pretrained(
        model_path=params["model_path"], **init_kwargs
    )
    generator.generate_video(prompt, **generation_kwargs)
    generator.shutdown()

    generated_video_path = os.path.join(output_dir, output_video_name)
    assert os.path.exists(generated_video_path), (
        f"Output video was not generated at {generated_video_path}"
    )

    # Find reference video
    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )
    if not os.path.exists(reference_folder):
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}"
        )

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        raise FileNotFoundError(
            f"Reference video not found for prompt: {prompt[:50]}... with backend: {ATTENTION_BACKEND}"
        )

    reference_video_path = os.path.join(reference_folder, reference_video_name)

    logger.info(f"Computing SSIM between {reference_video_path} and {generated_video_path}")
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path, generated_video_path, use_ms_ssim=True
    )

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")

    write_ssim_results(
        output_dir, ssim_values, reference_video_path, generated_video_path,
        params["num_inference_steps"], prompt
    )

    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}"
    )


@pytest.mark.parametrize("prompt", I2V_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
def test_longcat_i2v_similarity(prompt: str, ATTENTION_BACKEND: str):
    """
    Test LongCat I2V inference and compare output to reference videos using SSIM.
    
    Parameters derived from examples/inference/basic/basic_longcat_i2v.py
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    params = select_ssim_params(LONGCAT_I2V_PARAMS, LONGCAT_I2V_FULL_QUALITY_PARAMS)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "LongCat-Video-I2V"
    
    output_dir = build_generated_output_dir(
        script_dir,
        model_id,
        ATTENTION_BACKEND,
    )
    output_video_name = f"{prompt[:100].strip()}.mp4"
    os.makedirs(output_dir, exist_ok=True)

    # Get image path for this prompt
    prompt_idx = I2V_TEST_PROMPTS.index(prompt)
    image_path = _resolve_asset_path(I2V_IMAGE_PATHS[prompt_idx])

    init_kwargs = {
        "num_gpus": params["num_gpus"],
        "use_fsdp_inference": True,
        "dit_cpu_offload": True,
        "vae_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "enable_bsa": False,
    }

    generation_kwargs = {
        "output_path": output_dir,
        "image_path": image_path,
        "height": params["height"],
        "width": params["width"],
        "num_frames": params["num_frames"],
        "num_inference_steps": params["num_inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "fps": params["fps"],
        "seed": params["seed"],
        "negative_prompt": params["negative_prompt"],
    }

    generator = VideoGenerator.from_pretrained(
        model_path=params["model_path"], **init_kwargs
    )
    generator.generate_video(prompt, **generation_kwargs)
    generator.shutdown()

    generated_video_path = os.path.join(output_dir, output_video_name)
    assert os.path.exists(generated_video_path), (
        f"Output video was not generated at {generated_video_path}"
    )

    # Find reference video
    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )
    if not os.path.exists(reference_folder):
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}"
        )

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        raise FileNotFoundError(
            f"Reference video not found for prompt: {prompt[:50]}... with backend: {ATTENTION_BACKEND}"
        )

    reference_video_path = os.path.join(reference_folder, reference_video_name)

    logger.info(f"Computing SSIM between {reference_video_path} and {generated_video_path}")
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path, generated_video_path, use_ms_ssim=True
    )

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")

    write_ssim_results(
        output_dir, ssim_values, reference_video_path, generated_video_path,
        params["num_inference_steps"], prompt
    )

    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}"
    )


@pytest.mark.parametrize("prompt", VC_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
def test_longcat_vc_similarity(prompt: str, ATTENTION_BACKEND: str):
    """
    Test LongCat VC (Video Continuation) inference and compare output to reference videos using SSIM.
    
    Parameters derived from examples/inference/basic/basic_longcat_vc.py
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    params = select_ssim_params(LONGCAT_VC_PARAMS, LONGCAT_VC_FULL_QUALITY_PARAMS)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "LongCat-Video-VC"
    
    output_dir = build_generated_output_dir(
        script_dir,
        model_id,
        ATTENTION_BACKEND,
    )
    output_video_name = f"{prompt[:100].strip()}.mp4"
    os.makedirs(output_dir, exist_ok=True)

    # Get video path for this prompt
    prompt_idx = VC_TEST_PROMPTS.index(prompt)
    video_path = _resolve_asset_path(VC_VIDEO_PATHS[prompt_idx])
    
    if not os.path.exists(video_path):
        pytest.skip(f"Input video not found at {video_path}")

    init_kwargs = {
        "num_gpus": params["num_gpus"],
        "use_fsdp_inference": False,
        "dit_cpu_offload": False,
        "vae_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "pin_cpu_memory": False,
        "enable_bsa": False,
    }

    generation_kwargs = {
        "output_path": output_dir,
        "video_path": video_path,
        "num_cond_frames": params["num_cond_frames"],
        "height": params["height"],
        "width": params["width"],
        "num_frames": params["num_frames"],
        "num_inference_steps": params["num_inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "fps": params["fps"],
        "seed": params["seed"],
        "negative_prompt": params["negative_prompt"],
    }

    generator = VideoGenerator.from_pretrained(
        model_path=params["model_path"], **init_kwargs
    )
    generator.generate_video(prompt, **generation_kwargs)
    generator.shutdown()

    generated_video_path = os.path.join(output_dir, output_video_name)
    assert os.path.exists(generated_video_path), (
        f"Output video was not generated at {generated_video_path}"
    )

    # Find reference video
    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )
    if not os.path.exists(reference_folder):
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}"
        )

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        raise FileNotFoundError(
            f"Reference video not found for prompt: {prompt[:50]}... with backend: {ATTENTION_BACKEND}"
        )

    reference_video_path = os.path.join(reference_folder, reference_video_name)

    logger.info(f"Computing SSIM between {reference_video_path} and {generated_video_path}")
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path, generated_video_path, use_ms_ssim=True
    )

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")

    write_ssim_results(
        output_dir, ssim_values, reference_video_path, generated_video_path,
        params["num_inference_steps"], prompt
    )

    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}"
    )
