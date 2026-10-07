# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    run_text_to_video_similarity_test, )
from fastvideo.tests.ssim.reference_utils import (
    get_cuda_device_name,
    resolve_device_reference_folder,
    with_model_path,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H100", "H100"),
        ("GB200", "GB200"),
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

# The test adds "model_path" from FASTVIDEO_TEST_SD35_MODEL_DIR.
SD35_PARAMS = {
    "num_gpus": 1,
    "sp_size": 1,
    "tp_size": 1,
    "height": 256,
    "width": 256,
    "num_frames": 1,
    "fps": 1,
    "num_inference_steps": 8,
    "guidance_scale": 6.0,
    "seed": 0,
    "neg_prompt": "",
}


def _sd35_full_quality_params(model_path: str) -> dict[str, object]:
    """Return the full-quality SD3.5 params, taken from the default sampling params of ``model_path``."""
    full_quality_defaults = SamplingParam.from_pretrained(model_path)
    return {
        "num_gpus": 1,
        "model_path": model_path,
        "sp_size": 1,
        "tp_size": 1,
        "height": full_quality_defaults.height,
        "width": full_quality_defaults.width,
        "num_frames": SD35_PARAMS["num_frames"],
        "fps": full_quality_defaults.fps,
        "num_inference_steps": (full_quality_defaults.num_inference_steps),
        "guidance_scale": full_quality_defaults.guidance_scale,
        "seed": full_quality_defaults.seed,
        "neg_prompt": full_quality_defaults.negative_prompt,
    }


SD35_MODEL_TO_PARAMS = {
    MODEL_ID: SD35_PARAMS,
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
    sd35_model_path = envs.FASTVIDEO_TEST_SD35_MODEL_DIR.get()
    is_hf_repo = "/" in sd35_model_path and not sd35_model_path.startswith("/")
    if not is_hf_repo and not os.path.isdir(sd35_model_path):
        pytest.skip(f"SD3.5 weights not found at {sd35_model_path} (set FASTVIDEO_TEST_SD35_MODEL_DIR to override)")

    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=with_model_path(SD35_MODEL_TO_PARAMS, sd35_model_path),
        full_quality_params_map={model_id: _sd35_full_quality_params(sd35_model_path)},
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
