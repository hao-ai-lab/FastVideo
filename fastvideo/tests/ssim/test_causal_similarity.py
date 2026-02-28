# SPDX-License-Identifier: Apache-2.0
import json
import os

import torch
import pytest

from fastvideo import VideoGenerator
from fastvideo.configs.sample.wan import (
    SelfForcingWan2_1_T2V_1_3B_480P_SamplingParam,
)
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    build_reference_folder_path,
    get_cuda_device_name,
    resolve_device_reference_folder,
    select_ssim_params,
)
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H200", "H200"),
    ),
    device_name=get_cuda_device_name(),
    logger=logger,
)

# Base parameters from the shell script

SF_WAN_T2V_PARAMS = {
    "num_gpus": 1,
    "model_path": "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "num_inference_steps": 4,
    "seed": 1024,
    "sp_size": 1,
    "tp_size": 1,
}

_SF_WAN_T2V_FULL_QUALITY_DEFAULTS = (
    SelfForcingWan2_1_T2V_1_3B_480P_SamplingParam()
)
SF_WAN_T2V_FULL_QUALITY_PARAMS = {
    "num_gpus": SF_WAN_T2V_PARAMS["num_gpus"],
    "model_path": SF_WAN_T2V_PARAMS["model_path"],
    "height": _SF_WAN_T2V_FULL_QUALITY_DEFAULTS.height,
    "width": _SF_WAN_T2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": SF_WAN_T2V_PARAMS["num_frames"],  # default num_frames: 81
    "num_inference_steps": _SF_WAN_T2V_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _SF_WAN_T2V_FULL_QUALITY_DEFAULTS.guidance_scale,
    "seed": _SF_WAN_T2V_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": SF_WAN_T2V_PARAMS["sp_size"],
    "tp_size": SF_WAN_T2V_PARAMS["tp_size"],
    "neg_prompt": _SF_WAN_T2V_FULL_QUALITY_DEFAULTS.negative_prompt,
}


MODEL_TO_PARAMS = {
    "SFWan2.1-T2V-1.3B-Diffusers": SF_WAN_T2V_PARAMS,
}
FULL_QUALITY_MODEL_TO_PARAMS = {
    "SFWan2.1-T2V-1.3B-Diffusers": SF_WAN_T2V_FULL_QUALITY_PARAMS,
}

I2V_MODEL_TO_PARAMS = {
}

TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
    # "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature."
]

I2V_TEST_PROMPTS = [
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
]

I2V_IMAGE_PATHS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
]

@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_causal_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Test that runs inference with different parameters and compares the output
    to reference videos using SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = build_generated_output_dir(
        script_dir,
        model_id,
        ATTENTION_BACKEND,
    )
    output_video_name = f"{prompt[:100].strip()}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    params_map = select_ssim_params(
        MODEL_TO_PARAMS,
        FULL_QUALITY_MODEL_TO_PARAMS,
    )
    BASE_PARAMS = params_map[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "sp_size": BASE_PARAMS["sp_size"],
        "tp_size": BASE_PARAMS["tp_size"],
        "dit_cpu_offload": True,
    }
    if BASE_PARAMS.get("vae_sp"):
        init_kwargs["vae_sp"] = True
        init_kwargs["vae_tiling"] = True
    #if "text-encoder-precision" in BASE_PARAMS:
    #    init_kwargs["text_encoder_precisions"] = BASE_PARAMS["text-encoder-precision"]

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "seed": BASE_PARAMS["seed"],
    }
    if "neg_prompt" in BASE_PARAMS:
        generation_kwargs["neg_prompt"] = BASE_PARAMS["neg_prompt"]

    generator = VideoGenerator.from_pretrained(model_path=BASE_PARAMS["model_path"], **init_kwargs)
    generator.generate_video(prompt, **generation_kwargs)

    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

    assert os.path.exists(
        output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}")

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(f"Reference video not found for prompt: {prompt} with backend: {ATTENTION_BACKEND}")
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                 generated_video_path,
                                                 use_ms_ssim=True)

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(output_dir, ssim_values, reference_video_path,
                                 generated_video_path, num_inference_steps,
                                 prompt)

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.98
    assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for {model_id} with backend {ATTENTION_BACKEND}"
