# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest
import torch

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

OVIS_MODEL_PATH = os.getenv("OVIS_WEIGHTS", "AIDC-AI/Ovis-Image-7B")

device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H100", "H100"),
        ("H200", "H200"),
    ),
    device_name=get_cuda_device_name(),
    fallback_device_prefix="L40S",
    logger=logger,
)

MODEL_ID = "AIDC-AI__Ovis-Image-7B"

TEST_PROMPTS = [
    "A vibrant travel poster with the bold word EXPLORE in large clean sans-serif "
    "letters above a sunny mountain landscape, professional layout, high contrast, "
    "sharp typography, 4k quality",
]

OVIS_PARAMS = {
    "num_gpus": 1,
    "model_path": OVIS_MODEL_PATH,
    "sp_size": 1,
    "tp_size": 1,
    "height": 256,
    "width": 256,
    "num_frames": 1,
    "fps": 1,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": 42,
    "neg_prompt": "",
}

# Full-quality tier mirrors the official example (1024², 50 steps).
OVIS_FULL_QUALITY_PARAMS = {
    **OVIS_PARAMS,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
}

OVIS_MODEL_TO_PARAMS = {MODEL_ID: OVIS_PARAMS}
OVIS_FULL_QUALITY_MODEL_TO_PARAMS = {MODEL_ID: OVIS_FULL_QUALITY_PARAMS}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Ovis-Image SSIM test requires CUDA",
)
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(OVIS_MODEL_TO_PARAMS.keys()))
def test_ovis_image_similarity(
    prompt: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    is_hf_repo = "/" in OVIS_MODEL_PATH and not OVIS_MODEL_PATH.startswith("/")
    if not is_hf_repo and not os.path.isdir(OVIS_MODEL_PATH):
        pytest.skip(f"Ovis-Image weights not found at {OVIS_MODEL_PATH} "
                    "(set OVIS_WEIGHTS to override)")

    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=OVIS_MODEL_TO_PARAMS,
        full_quality_params_map=OVIS_FULL_QUALITY_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.98,
        init_kwargs_override={
            "use_fsdp_inference": False,
            "text_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "pin_cpu_memory": False,
        },
        generation_kwargs_override={
            "save_video": True,
        },
    )
