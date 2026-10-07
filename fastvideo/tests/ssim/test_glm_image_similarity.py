# SPDX-License-Identifier: Apache-2.0
"""SSIM-based regression test for GLM-Image generation."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import fastvideo.envs as envs
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
pytestmark = pytest.mark.skip(reason="Disabled pending removal of GLM-Image support.")

REPO_ROOT = Path(__file__).resolve().parents[3]

device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H100", "H100"),
        ("H200", "H200"),
        # GB200 must precede B200: the lookup is a substring scan and
        # "B200" matches "NVIDIA GB200", so the coarser pattern would
        # otherwise claim every Grace-Blackwell host.
        ("GB200", "GB200"),
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

# The test adds "model_path" from FASTVIDEO_TEST_GLM_IMAGE_MODEL_DIR.
GLM_IMAGE_PARAMS = {
    "num_gpus": 1,
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


def _local_weights_dir() -> Path:
    return Path(envs.FASTVIDEO_TEST_GLM_IMAGE_LOCAL_WEIGHTS_DIR.get() or REPO_ROOT / "official_weights" / "glm_image")


def _has_weights() -> bool:
    required = ["transformer", "vae", "text_encoder", "vision_language_encoder", "processor", "tokenizer", "scheduler"]
    return all((_local_weights_dir() / r).exists() for r in required)


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
    if not _has_weights():
        pytest.skip(f"GLM-Image full weights not found at {_local_weights_dir()}.")
    model_path = envs.FASTVIDEO_TEST_GLM_IMAGE_MODEL_DIR.get() or str(_local_weights_dir())

    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=with_model_path(GLM_IMAGE_MODEL_TO_PARAMS, model_path),
        full_quality_params_map=with_model_path(GLM_IMAGE_FULL_QUALITY_MODEL_TO_PARAMS, model_path),
        min_acceptable_ssim=0.98,
        init_kwargs_override={
            "trust_remote_code": True,
            "use_fsdp_inference": False,
        },
        generation_kwargs_override={
            "save_video": True,
        },
    )
