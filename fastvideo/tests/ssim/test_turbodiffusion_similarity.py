# SPDX-License-Identifier: Apache-2.0
"""
SSIM-based similarity test for TurboDiffusion inference.

TurboDiffusion uses the SLA (Sparse-Linear Attention) backend and RCM scheduler
for 1-4 step video generation.
"""
import os

import torch
import pytest

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)

device_name = torch.cuda.get_device_name()
device_reference_folder_suffix = '_reference_videos'

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
else:
    # device_reference_folder = "L40S" + device_reference_folder_suffix
    logger.warning(f"Unsupported device for ssim tests: {device_name}, using L40S references")
    raise ValueError(f"Unsupported device for ssim tests: {device_name}")


# TurboDiffusion parameters (1-4 step generation with RCM scheduler + SLA attention)
TURBODIFFUSION_PARAMS = {
    "num_gpus": 2,
    "model_path": "loayrashid/TurboWan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "num_inference_steps": 4,  # TurboDiffusion uses 1-4 steps
    "guidance_scale": 1.0,  # No CFG for TurboDiffusion
    "seed": 42,
    "sp_size": 2,
    "tp_size": 1,
    "fps": 24,
}

TURBODIFFUSION_MODEL_TO_PARAMS = {
    "TurboWan2.1-T2V-1.3B-Diffusers": TURBODIFFUSION_PARAMS,
}

TURBODIFFUSION_TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
]


@pytest.mark.parametrize("prompt", TURBODIFFUSION_TEST_PROMPTS)
@pytest.mark.parametrize("model_id", list(TURBODIFFUSION_MODEL_TO_PARAMS.keys()))
def test_turbodiffusion_inference_similarity(prompt, model_id):
    """
    Test that runs TurboDiffusion inference with SLA attention and RCM scheduler,
    then compares the output to reference videos using SSIM.
    """
    # TurboDiffusion requires SLA attention backend
    ATTENTION_BACKEND = "SLA_ATTN"
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_output_dir = os.path.join(script_dir, 'generated_videos', model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip()}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = TURBODIFFUSION_MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "sp_size": BASE_PARAMS["sp_size"],
        "tp_size": BASE_PARAMS["tp_size"],
        "override_pipeline_cls_name": "TurboDiffusionPipeline",
    }

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
    }

    generator = VideoGenerator.from_pretrained(
        model_path=BASE_PARAMS["model_path"],
        **init_kwargs
    )
    generator.generate_video(prompt, **generation_kwargs)

    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

    assert os.path.exists(output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = os.path.join(
        script_dir, device_reference_folder, model_id, ATTENTION_BACKEND
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}"
        )

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(
            f"Reference video not found for prompt: {prompt} with backend: {ATTENTION_BACKEND}"
        )
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path, generated_video_path, use_ms_ssim=True
    )

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(
        output_dir, ssim_values, reference_video_path,
        generated_video_path, num_inference_steps, prompt
    )

    if not success:
        logger.error("Failed to write SSIM results to file")

    # TurboDiffusion uses fewer steps, may have slightly lower SSIM
    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}"
    )


# TurboDiffusion I2V parameters (dual-model with RCM scheduler + SLA attention)
TURBODIFFUSION_I2V_PARAMS = {
    "num_gpus": 2,
    "model_path": "loayrashid/TurboWan2.2-I2V-A14B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 4,  # TurboDiffusion uses 1-4 steps
    "guidance_scale": 1.0,  # No CFG for TurboDiffusion
    "seed": 42,
    "sp_size": 2,
    "tp_size": 1,
    "fps": 24,
}

TURBODIFFUSION_I2V_MODEL_TO_PARAMS = {
    "TurboWan2.2-I2V-A14B-Diffusers": TURBODIFFUSION_I2V_PARAMS,
}

TURBODIFFUSION_I2V_TEST_PROMPTS = [
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
]

TURBODIFFUSION_I2V_IMAGE_PATHS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
]


@pytest.mark.parametrize("prompt", TURBODIFFUSION_I2V_TEST_PROMPTS)
@pytest.mark.parametrize("model_id", list(TURBODIFFUSION_I2V_MODEL_TO_PARAMS.keys()))
def test_turbodiffusion_i2v_inference_similarity(prompt, model_id):
    """
    Test that runs TurboDiffusion I2V inference with dual-model switching,
    then compares the output to reference videos using SSIM.
    """
    # TurboDiffusion requires SLA attention backend
    ATTENTION_BACKEND = "SLA_ATTN"
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    assert len(TURBODIFFUSION_I2V_TEST_PROMPTS) == len(TURBODIFFUSION_I2V_IMAGE_PATHS), \
        "Expect number of prompts equal to number of images"

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_output_dir = os.path.join(script_dir, 'generated_videos', model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip()}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = TURBODIFFUSION_I2V_MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]
    image_path = TURBODIFFUSION_I2V_IMAGE_PATHS[TURBODIFFUSION_I2V_TEST_PROMPTS.index(prompt)]

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "sp_size": BASE_PARAMS["sp_size"],
        "tp_size": BASE_PARAMS["tp_size"],
        "override_pipeline_cls_name": "TurboDiffusionI2VPipeline",
    }

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "image_path": image_path,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
    }

    generator = VideoGenerator.from_pretrained(
        model_path=BASE_PARAMS["model_path"],
        **init_kwargs
    )
    generator.generate_video(prompt, **generation_kwargs)

    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

    assert os.path.exists(output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = os.path.join(
        script_dir, device_reference_folder, model_id, ATTENTION_BACKEND
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}"
        )

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(
            f"Reference video not found for prompt: {prompt} with backend: {ATTENTION_BACKEND}"
        )
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path, generated_video_path, use_ms_ssim=True
    )

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(
        output_dir, ssim_values, reference_video_path,
        generated_video_path, num_inference_steps, prompt
    )

    if not success:
        logger.error("Failed to write SSIM results to file")

    # TurboDiffusion I2V uses fewer steps, may have slightly lower SSIM
    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}"
    )
