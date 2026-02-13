# SPDX-License-Identifier: Apache-2.0
"""
SSIM regression test for HunyuanGameCraft (T2V and I2V).

Generates a video with deterministic seed and camera trajectory,
then compares against a device-specific reference video via MS-SSIM.

Reference videos must be pre-generated and stored under:
    <device>_reference_videos/HunyuanGameCraft/<ATTENTION_BACKEND>/

To create initial reference videos, run this test once and copy the
generated videos into the appropriate reference folder.
"""
import os

import torch
import pytest

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.utils import (
    compute_video_ssim_torchvision,
    write_ssim_results,
)
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Device-dependent reference folder
# ---------------------------------------------------------------------------
device_name = torch.cuda.get_device_name()
device_reference_folder_suffix = "_reference_videos"

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
elif "A100" in device_name:
    device_reference_folder = "A100" + device_reference_folder_suffix
elif "H100" in device_name:
    device_reference_folder = "H100" + device_reference_folder_suffix
else:
    device_reference_folder = "Unknown" + device_reference_folder_suffix
    logger.warning(f"Unsupported device for ssim tests: {device_name}")

# ---------------------------------------------------------------------------
# Helpers – camera trajectory (self-contained, no official repo dependency)
# ---------------------------------------------------------------------------
from fastvideo.models.camera import create_camera_trajectory as _create_camera_trajectory


def _shutdown_executor(generator: VideoGenerator | None) -> None:
    if generator is None:
        return
    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
GAMECRAFT_T2V_PARAMS = {
    "num_gpus": 1,
    "model_path": "official_weights/hunyuan-gamecraft-diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 33,
    "num_inference_steps": 20,
    "guidance_scale": 6.0,
    "seed": 1024,
    "action": "forward",
    "action_speed": 0.2,
    "negative_prompt": "",
}

GAMECRAFT_I2V_PARAMS = {
    **GAMECRAFT_T2V_PARAMS,
    "image_path": (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/diffusers/astronaut.jpg"
    ),
}

MODEL_TO_PARAMS = {
    "HunyuanGameCraft-T2V": GAMECRAFT_T2V_PARAMS,
}

I2V_MODEL_TO_PARAMS = {
    "HunyuanGameCraft-I2V": GAMECRAFT_I2V_PARAMS,
}

TEST_PROMPTS = [
    "A majestic ancient temple stands under a clear blue sky, its grandeur highlighted by towering Doric columns and intricate architectural details.",
]

I2V_TEST_PROMPTS = [
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background.",
]

# ---------------------------------------------------------------------------
# T2V SSIM test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_gamecraft_t2v_similarity(prompt, ATTENTION_BACKEND, model_id):
    """Generate a T2V video with GameCraft and compare to reference via SSIM."""
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, "generated_videos", model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip()}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]

    # Build camera trajectory
    camera_states = _create_camera_trajectory(
        action=BASE_PARAMS["action"],
        height=BASE_PARAMS["height"],
        width=BASE_PARAMS["width"],
        num_frames=BASE_PARAMS["num_frames"],
        action_speed=BASE_PARAMS["action_speed"],
        dtype=torch.bfloat16,
    )

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "use_fsdp_inference": True,
        "dit_cpu_offload": True,
        "vae_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "pin_cpu_memory": True,
    }

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": 24,
        "camera_states": camera_states,
        "negative_prompt": BASE_PARAMS.get("negative_prompt", ""),
        "save_video": True,
    }

    generator: VideoGenerator | None = None
    try:
        generator = VideoGenerator.from_pretrained(
            model_path=BASE_PARAMS["model_path"], **init_kwargs
        )
        generator.generate_video(prompt, **generation_kwargs)
    finally:
        _shutdown_executor(generator)

    assert os.path.exists(output_dir), (
        f"Output video was not generated at {output_dir}"
    )

    # Compare to reference
    reference_folder = os.path.join(
        script_dir, device_reference_folder, model_id, ATTENTION_BACKEND
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}"
        )

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(
            f"Reference video not found for prompt: {prompt} "
            f"with backend: {ATTENTION_BACKEND}"
        )
        raise FileNotFoundError("Reference video missing")

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
        output_dir,
        ssim_values,
        reference_video_path,
        generated_video_path,
        num_inference_steps,
        prompt,
    )

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.93
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}"
    )


# ---------------------------------------------------------------------------
# I2V SSIM test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", I2V_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(I2V_MODEL_TO_PARAMS.keys()))
def test_gamecraft_i2v_similarity(prompt, ATTENTION_BACKEND, model_id):
    """Generate an I2V video with GameCraft and compare to reference via SSIM."""
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, "generated_videos", model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip()}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = I2V_MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]

    # Build camera trajectory
    camera_states = _create_camera_trajectory(
        action=BASE_PARAMS["action"],
        height=BASE_PARAMS["height"],
        width=BASE_PARAMS["width"],
        num_frames=BASE_PARAMS["num_frames"],
        action_speed=BASE_PARAMS["action_speed"],
        dtype=torch.bfloat16,
    )

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "use_fsdp_inference": True,
        "dit_cpu_offload": True,
        "vae_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "pin_cpu_memory": True,
    }

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "image_path": BASE_PARAMS["image_path"],
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": 24,
        "camera_states": camera_states,
        "negative_prompt": BASE_PARAMS.get("negative_prompt", ""),
        "save_video": True,
    }

    generator: VideoGenerator | None = None
    try:
        generator = VideoGenerator.from_pretrained(
            model_path=BASE_PARAMS["model_path"], **init_kwargs
        )
        generator.generate_video(prompt, **generation_kwargs)
    finally:
        _shutdown_executor(generator)

    assert os.path.exists(output_dir), (
        f"Output video was not generated at {output_dir}"
    )

    # Compare to reference
    reference_folder = os.path.join(
        script_dir, device_reference_folder, model_id, ATTENTION_BACKEND
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}"
        )

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(
            f"Reference video not found for prompt: {prompt} "
            f"with backend: {ATTENTION_BACKEND}"
        )
        raise FileNotFoundError("Reference video missing")

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
        output_dir,
        ssim_values,
        reference_video_path,
        generated_video_path,
        num_inference_steps,
        prompt,
    )

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.93
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}"
    )
