# SPDX-License-Identifier: Apache-2.0
"""SSIM-based regression test for GLM-Image generation."""
from __future__ import annotations

import os
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_WEIGHTS_DIR = Path(
    os.getenv("GLM_IMAGE_LOCAL_WEIGHTS_DIR",
              REPO_ROOT / "official_weights" / "glm_image"))
GLM_IMAGE_MODEL_PATH = os.getenv("GLM_IMAGE_MODEL_DIR", str(LOCAL_WEIGHTS_DIR))

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

MODEL_ID = "zai-org__GLM-Image"

TEST_PROMPTS = [
    "A beautiful landscape photography with rolling hills, "
    "a winding river, and a vibrant sunset in the background. "
    "Warm golden light, photorealistic style.",
]

GLM_IMAGE_PARAMS = {
    "num_gpus": 1,
    "model_path": GLM_IMAGE_MODEL_PATH,
    "sp_size": 1,
    "tp_size": 1,
    "height": 256,
    "width": 256,
    "num_frames": 1,
    "fps": 1,
    "num_inference_steps": 4,
    "guidance_scale": 1.5,
    "seed": 0,
    "neg_prompt": "",
}

GLM_IMAGE_FULL_QUALITY_PARAMS = {
    "num_gpus": 1,
    "model_path": GLM_IMAGE_MODEL_PATH,
    "sp_size": 1,
    "tp_size": 1,
    "height": 1024,
    "width": 1024,
    "num_frames": 1,
    "fps": 1,
    "num_inference_steps": 50,
    "guidance_scale": 1.5,
    "seed": 0,
    "neg_prompt": "",
}

GLM_IMAGE_MODEL_TO_PARAMS = {
    MODEL_ID: GLM_IMAGE_PARAMS,
}
GLM_IMAGE_FULL_QUALITY_MODEL_TO_PARAMS = {
    MODEL_ID: GLM_IMAGE_FULL_QUALITY_PARAMS,
}


def _has_weights() -> bool:
    required = ["transformer", "vae", "text_encoder",
                "vision_language_encoder", "processor", "tokenizer",
                "scheduler"]
    return all((LOCAL_WEIGHTS_DIR / r).exists() for r in required)


def _upstream_glm_image_available() -> bool:
    try:
        import transformers
    except ImportError:
        return False
    return hasattr(transformers, "GlmImageForConditionalGeneration")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GLM-Image SSIM test requires CUDA",
)
@pytest.mark.skipif(
    not _has_weights(),
    reason=f"GLM-Image full weights not found at {LOCAL_WEIGHTS_DIR}.",
)
@pytest.mark.skipif(
    not _upstream_glm_image_available(),
    reason="GLM-Image needs transformers>=5.0.0rc0 (ships the AR encoder).",
)
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(GLM_IMAGE_MODEL_TO_PARAMS.keys()))
def test_glm_image_similarity(
    prompt: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=GLM_IMAGE_MODEL_TO_PARAMS,
        full_quality_params_map=GLM_IMAGE_FULL_QUALITY_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.98,
        init_kwargs_override={
            "trust_remote_code": True,
            "use_fsdp_inference": False,
        },
        generation_kwargs_override={
            "save_video": True,
        },
    )
