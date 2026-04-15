# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest
import torch

from fastvideo.configs.sample.base import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    run_text_to_video_similarity_test,
)
from fastvideo.tests.ssim.reference_utils import (
    get_cuda_device_name,
    resolve_device_reference_folder,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

SD35_MODEL_PATH = os.getenv(
    "SD35_MODEL_DIR",
    "stabilityai/stable-diffusion-3.5-medium",
)

device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H100", "H100"),
        ("H200", "H200"),
        ("RTX 4090", "RTX4090"),
        ("4090", "RTX4090"),
    ),
    device_name=get_cuda_device_name(),
    fallback_device_prefix="L40S",
    logger=logger,
)

MODEL_ID = "stabilityai__stable-diffusion-3.5-medium"

TEST_PROMPTS = [
    "a photo of a cat",
]

SD35_PARAMS = {
    "num_gpus": 1,
    "model_path": SD35_MODEL_PATH,
    "sp_size": 1,
    "tp_size": 1,
    "height": 256,
    "width": 256,
    "num_frames": 1,
    "fps": 1,
    "num_inference_steps": 8,
    "guidance_scale": 6.0,
    "seed": 0,
    "neg_prompt": "lowres, blurry",
}

_SD35_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium")
SD35_FULL_QUALITY_PARAMS = {
    "num_gpus": 1,
    "model_path": SD35_MODEL_PATH,
    "sp_size": 1,
    "tp_size": 1,
    "height": _SD35_FULL_QUALITY_DEFAULTS.height,
    "width": _SD35_FULL_QUALITY_DEFAULTS.width,
    "num_frames": SD35_PARAMS["num_frames"],
    "fps": _SD35_FULL_QUALITY_DEFAULTS.fps,
    "num_inference_steps": (_SD35_FULL_QUALITY_DEFAULTS.num_inference_steps),
    "guidance_scale": _SD35_FULL_QUALITY_DEFAULTS.guidance_scale,
    "seed": _SD35_FULL_QUALITY_DEFAULTS.seed,
    "neg_prompt": _SD35_FULL_QUALITY_DEFAULTS.negative_prompt,
}

SD35_MODEL_TO_PARAMS = {
    MODEL_ID: SD35_PARAMS,
}
SD35_FULL_QUALITY_MODEL_TO_PARAMS = {
    MODEL_ID: SD35_FULL_QUALITY_PARAMS,
}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SD3.5 SSIM test requires CUDA",
)
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(SD35_MODEL_TO_PARAMS.keys()))
def test_sd35_similarity(
    prompt: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    is_hf_repo = "/" in SD35_MODEL_PATH and not SD35_MODEL_PATH.startswith("/")
    if not is_hf_repo and not os.path.isdir(SD35_MODEL_PATH):
        pytest.skip(f"SD3.5 weights not found at {SD35_MODEL_PATH} (set SD35_MODEL_DIR to override)")

    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=SD35_MODEL_TO_PARAMS,
        full_quality_params_map=SD35_FULL_QUALITY_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.98,
        init_kwargs_override={
            "workload_type": "t2v",
            "use_fsdp_inference": False,
            "text_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "image_encoder_cpu_offload": False,
            "pin_cpu_memory": False,
        },
        generation_kwargs_override={
            "save_video": True,
        },
    )
