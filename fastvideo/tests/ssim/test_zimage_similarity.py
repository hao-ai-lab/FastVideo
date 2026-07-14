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
# Calibrate after the first repeated L40S reference runs.
ZIMAGE_MIN_SSIM = 0.98

ZIMAGE_MODEL_PATH = os.getenv(
    "ZIMAGE_MODEL_DIR",
    "Tongyi-MAI/Z-Image-Turbo",
)

TEST_PROMPTS = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
]

device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H100", "H100"),
        ("H200", "H200"),
        ("B200", "B200"),
    ),
    device_name=get_cuda_device_name(),
    fallback_device_prefix="L40S",
    logger=logger,
)

ZIMAGE_MODEL_TO_PARAMS: dict[str, dict[str, object]] = {
    "Tongyi-MAI__Z-Image-Turbo": {
        "num_gpus": 1,
        "model_path": ZIMAGE_MODEL_PATH,
        "sp_size": 1,
        "tp_size": 1,
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "num_inference_steps": 9,
        "guidance_scale": 0.0,
        "seed": 42,
    },
}

ZIMAGE_FULL_QUALITY_MODEL_TO_PARAMS: dict[str, dict[str, object]] = {
    "Tongyi-MAI__Z-Image-Turbo": dict(
        ZIMAGE_MODEL_TO_PARAMS["Tongyi-MAI__Z-Image-Turbo"]
    ),
}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Z-Image SSIM test requires CUDA",
)
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(ZIMAGE_MODEL_TO_PARAMS))
def test_zimage_similarity(
    prompt: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    is_hf_repo = "/" in ZIMAGE_MODEL_PATH and not ZIMAGE_MODEL_PATH.startswith("/")
    if not is_hf_repo and not os.path.isdir(ZIMAGE_MODEL_PATH):
        pytest.skip(
            f"Z-Image weights not found at {ZIMAGE_MODEL_PATH} "
            "(set ZIMAGE_MODEL_DIR to override)"
        )

    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=ZIMAGE_MODEL_TO_PARAMS,
        full_quality_params_map=ZIMAGE_FULL_QUALITY_MODEL_TO_PARAMS,
        min_acceptable_ssim=ZIMAGE_MIN_SSIM,
        media_extension=".png",
        init_kwargs_override={
            "workload_type": "t2i",
            "use_fsdp_inference": False,
        },
        generation_kwargs_override={
            "save_video": True,
        },
    )
