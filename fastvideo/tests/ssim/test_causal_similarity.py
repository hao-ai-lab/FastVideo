# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
    run_text_to_video_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_inference_device_reference_folder(logger)

SF_WAN_T2V_PARAMS = {
    "num_gpus": 1,
    "model_path": "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "num_inference_steps": 4,
    "seed": 1024,
    "sp_size": 1,
    "tp_size": 1,
}

_SF_WAN_T2V_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers")
SF_WAN_T2V_FULL_QUALITY_PARAMS = {
    "num_gpus": SF_WAN_T2V_PARAMS["num_gpus"],
    "model_path": SF_WAN_T2V_PARAMS["model_path"],
    "height": _SF_WAN_T2V_FULL_QUALITY_DEFAULTS.height,
    "width": _SF_WAN_T2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": SF_WAN_T2V_PARAMS["num_frames"],
    "num_inference_steps": (_SF_WAN_T2V_FULL_QUALITY_DEFAULTS.num_inference_steps),
    "guidance_scale": (_SF_WAN_T2V_FULL_QUALITY_DEFAULTS.guidance_scale),
    "seed": _SF_WAN_T2V_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": SF_WAN_T2V_PARAMS["sp_size"],
    "tp_size": SF_WAN_T2V_PARAMS["tp_size"],
    "neg_prompt": (_SF_WAN_T2V_FULL_QUALITY_DEFAULTS.negative_prompt),
}

MODEL_TO_PARAMS = {
    "SFWan2.1-T2V-1.3B-Diffusers": SF_WAN_T2V_PARAMS,
}
FULL_QUALITY_MODEL_TO_PARAMS = {
    "SFWan2.1-T2V-1.3B-Diffusers": SF_WAN_T2V_FULL_QUALITY_PARAMS,
}

TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting "
    "with the energetic background of a bustling street food market. "
    "The scene captures a mix of humor and authenticity. Mid-shot "
    "framing, vibrant lighting.",
]


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_causal_similarity(
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
        default_params_map=MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.98,
        init_kwargs_override={"dit_cpu_offload": True},
    )
