# SPDX-License-Identifier: Apache-2.0
"""SSIM regression for Wan2.2-TI2V-5B.

One checkpoint serves two workloads, so there are two cases: text-to-video,
and image-to-video when an image is passed. For the image case the pipeline
ignores ``height``/``width`` and derives the output size from the image's
aspect ratio inside a hard-coded 480x832 pixel budget (see
``InputValidationStage``), so the sizes below bind only the text case.
"""
import os

import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
    run_image_to_video_similarity_test,
    run_text_to_video_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_inference_device_reference_folder(logger)

# Few steps and a short clip keep the CI run small; everything else (guidance,
# flow shift, fps, negative prompt) comes from the model's registered preset.
WAN_TI2V_PARAMS = {
    "num_gpus": 1,
    "model_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 4,
    "seed": 1024,
}
_WAN_TI2V_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(WAN_TI2V_PARAMS["model_path"])
WAN_TI2V_FULL_QUALITY_PARAMS = {
    "num_gpus": WAN_TI2V_PARAMS["num_gpus"],
    "model_path": WAN_TI2V_PARAMS["model_path"],
    "height": _WAN_TI2V_FULL_QUALITY_DEFAULTS.height,
    "width": _WAN_TI2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": WAN_TI2V_PARAMS["num_frames"],  # preset default is 121; 45 keeps the run in budget
    "num_inference_steps": _WAN_TI2V_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "seed": _WAN_TI2V_FULL_QUALITY_DEFAULTS.seed,
}

WAN_TI2V_MODEL_TO_PARAMS = {
    "Wan2.2-TI2V-5B-Diffusers": WAN_TI2V_PARAMS,
}
FULL_QUALITY_WAN_TI2V_MODEL_TO_PARAMS = {
    "Wan2.2-TI2V-5B-Diffusers": WAN_TI2V_FULL_QUALITY_PARAMS,
}

# Prompts and image are the ones in examples/inference/basic/basic_wan2_2_ti2v.py.
WAN_TI2V_T2V_TEST_PROMPTS = [
    ("A majestic lion strides across the golden savanna, its powerful frame glistening under the warm "
     "afternoon sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. "
     "The tone is vibrant, embodying the raw energy of the wild. Low angle, steady tracking shot, cinematic."),
]

WAN_TI2V_I2V_TEST_CASES = [
    (
        ("Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred "
         "feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the "
         "background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white "
         "clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. "
         "A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."),
        "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG",
    ),
]


@pytest.mark.parametrize("prompt", WAN_TI2V_T2V_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN", "TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(WAN_TI2V_MODEL_TO_PARAMS.keys()))
def test_wan_ti2v_t2v_inference_similarity(
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
        default_params_map=WAN_TI2V_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_WAN_TI2V_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.93,
    )


@pytest.mark.parametrize(("prompt", "image_path"), WAN_TI2V_I2V_TEST_CASES)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(WAN_TI2V_MODEL_TO_PARAMS.keys()))
def test_wan_ti2v_i2v_inference_similarity(
    prompt: str,
    image_path: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    run_image_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        image_path=image_path,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=WAN_TI2V_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_WAN_TI2V_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.97,
    )
