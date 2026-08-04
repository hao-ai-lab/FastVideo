# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
    run_text_to_video_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 4

device_reference_folder = resolve_inference_device_reference_folder(logger)

MINIMAX_H3_PARAMS = {
    "num_gpus": 4,
    "model_path": "MiniMaxAI/MiniMax-H3",
    "height": 768,
    "width": 1344,
    "num_frames": 124,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 0,
    "sp_size": 4,
    "tp_size": 1,
    "fps": 24,
    "neg_prompt": "",
}
MINIMAX_H3_FULL_QUALITY_PARAMS = {
    **MINIMAX_H3_PARAMS,
    "num_inference_steps": 50,
}

MINIMAX_H3_MODEL_TO_PARAMS = {
    "MiniMax-H3": MINIMAX_H3_PARAMS,
}
FULL_QUALITY_MINIMAX_H3_MODEL_TO_PARAMS = {
    "MiniMax-H3": MINIMAX_H3_FULL_QUALITY_PARAMS,
}

MINIMAX_H3_TEST_PROMPTS = [
    "A blue-white FastVideo logo forms from glowing particles in a black futuristic digital city",
]


@pytest.mark.parametrize("prompt", MINIMAX_H3_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(MINIMAX_H3_MODEL_TO_PARAMS))
def test_minimax_h3_inference_similarity(
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
        default_params_map=MINIMAX_H3_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_MINIMAX_H3_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.95,
    )
