# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.api.flux import FluxSamplingParam
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

# MS-SSIM gate (see module docstring).
FLUX_T2I_MIN_SSIM = 0.98

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

# Folder token must match Hub path with slashes → double underscore (SD3.5
# pattern in ``test_sd35_similarity.py``).
MODEL_ID = "black-forest-labs__FLUX.1-dev"

TEST_PROMPTS = [
    "a photo of a cat",
]

# The test adds "model_path" from FASTVIDEO_TEST_FLUX_T2I_MODEL_DIR.
FLUX_DEFAULT_PARAMS: dict[str, object] = {
    "num_gpus": 1,
    "sp_size": 1,
    "tp_size": 1,
    "height": 256,
    "width": 256,
    "num_frames": 1,
    "fps": 1,
    "num_inference_steps": 8,
    "guidance_scale": 3.5,
    "seed": 0,
}

_flux_full_defaults = FluxSamplingParam()
FLUX_FULL_QUALITY_PARAMS: dict[str, object] = {
    "num_gpus": 1,
    "sp_size": 1,
    "tp_size": 1,
    "height": _flux_full_defaults.height,
    "width": _flux_full_defaults.width,
    "num_frames": 1,
    "fps": _flux_full_defaults.fps,
    "num_inference_steps": _flux_full_defaults.num_inference_steps,
    "guidance_scale": _flux_full_defaults.guidance_scale,
    "seed": _flux_full_defaults.seed,
}

FLUX_MODEL_TO_PARAMS = {
    MODEL_ID: FLUX_DEFAULT_PARAMS,
}
FLUX_FULL_QUALITY_MODEL_TO_PARAMS = {
    MODEL_ID: FLUX_FULL_QUALITY_PARAMS,
}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FLUX T2I SSIM test requires CUDA",
)
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(FLUX_MODEL_TO_PARAMS.keys()))
def test_flux_t2i_similarity(
    prompt: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    flux_model_path = envs.FASTVIDEO_TEST_FLUX_T2I_MODEL_DIR.get()
    is_hf_repo = "/" in flux_model_path and not flux_model_path.startswith("/")
    if not is_hf_repo and not os.path.isdir(flux_model_path):
        pytest.skip(f"FLUX weights not found at {flux_model_path} "
                    f"(set FASTVIDEO_TEST_FLUX_T2I_MODEL_DIR to override)")

    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=with_model_path(FLUX_MODEL_TO_PARAMS, flux_model_path),
        full_quality_params_map=with_model_path(FLUX_FULL_QUALITY_MODEL_TO_PARAMS, flux_model_path),
        min_acceptable_ssim=FLUX_T2I_MIN_SSIM,
        media_extension=".png",
        init_kwargs_override={
            "workload_type": "t2i",
            "use_fsdp_inference": False,
            "text_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "image_encoder_cpu_offload": False,
            "pin_cpu_memory": False,
        },
        generation_kwargs_override={
            "save_video": True,
            "use_embedded_guidance": True,
            "true_cfg_scale": 1.0,
        },
    )
