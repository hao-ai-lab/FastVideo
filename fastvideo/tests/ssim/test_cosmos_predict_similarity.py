# SPDX-License-Identifier: Apache-2.0
import os
import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    DEVICE_MAPPINGS,
    resolve_inference_device_reference_folder,
    run_text_to_video_similarity_test,
)
from fastvideo.tests.ssim.reference_utils import (
    get_cuda_device_name,
    resolve_device_reference_folder,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_name = get_cuda_device_name()
device_reference_folder = resolve_device_reference_folder(
    DEVICE_MAPPINGS,
    device_name=device_name,
)
if device_reference_folder is None:
    raise ValueError(f"Unsupported device for ssim tests: {device_name}")

COSMOS_PREDICT_PARAMS = {
    "num_gpus": 1,
    "model_path": "nvidia/Cosmos-1.0-Prompt2World-7B-Video",
    "height": 128,
    "width": 128,
    "num_frames": 9,
    "num_inference_steps": 2,
    "guidance_scale": 7.0,
    "seed": 42,
    "sp_size": 1,
    "tp_size": 1,
    "fps": 24,
}

COSMOS_PREDICT_MODEL_TO_PARAMS = {
    "Cosmos-1.0-Prompt2World-7B-Video": COSMOS_PREDICT_PARAMS,
}

FULL_QUALITY_COSMOS_PREDICT_MODEL_TO_PARAMS = {
    "Cosmos-1.0-Prompt2World-7B-Video": COSMOS_PREDICT_PARAMS,
}

COSMOS_PREDICT_TEST_PROMPTS = [
    "A cute dog walking on grass.",
]

SSIM_THRESHOLD = 0.90

@pytest.mark.parametrize("prompt", COSMOS_PREDICT_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(COSMOS_PREDICT_MODEL_TO_PARAMS.keys()))
def test_cosmos_predict_inference_similarity(
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
        default_params_map=COSMOS_PREDICT_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_COSMOS_PREDICT_MODEL_TO_PARAMS,
        threshold=SSIM_THRESHOLD,
    )
