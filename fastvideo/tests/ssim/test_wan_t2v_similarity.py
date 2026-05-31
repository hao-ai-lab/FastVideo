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

REQUIRED_GPUS = 2

device_reference_folder = resolve_inference_device_reference_folder(logger)

WAN_T2V_PARAMS = {
    "num_gpus": 2,
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3,
    "embedded_cfg_scale": 6,
    "flow_shift": 7.0,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 1,
    "vae_sp": True,
    "fps": 24,
    "neg_prompt": (
        "Bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low "
        "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards"
    ),
    "text-encoder-precision": ("fp32",),
}
_WAN_T2V_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
WAN_T2V_FULL_QUALITY_PARAMS = {
    "num_gpus": WAN_T2V_PARAMS["num_gpus"],
    "model_path": WAN_T2V_PARAMS["model_path"],
    "height": _WAN_T2V_FULL_QUALITY_DEFAULTS.height,
    "width": _WAN_T2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": WAN_T2V_PARAMS["num_frames"],  # default num_frames: 81
    "num_inference_steps": _WAN_T2V_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _WAN_T2V_FULL_QUALITY_DEFAULTS.guidance_scale,
    "embedded_cfg_scale": WAN_T2V_PARAMS["embedded_cfg_scale"],
    "flow_shift": WAN_T2V_PARAMS["flow_shift"],
    "seed": _WAN_T2V_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": WAN_T2V_PARAMS["sp_size"],
    "tp_size": WAN_T2V_PARAMS["tp_size"],
    "vae_sp": WAN_T2V_PARAMS["vae_sp"],
    "fps": _WAN_T2V_FULL_QUALITY_DEFAULTS.fps,
    "neg_prompt": _WAN_T2V_FULL_QUALITY_DEFAULTS.negative_prompt,
    "text-encoder-precision": WAN_T2V_PARAMS["text-encoder-precision"],
}

WAN_T2V_MODEL_TO_PARAMS = {
    "Wan2.1-T2V-1.3B-Diffusers": WAN_T2V_PARAMS,
}
FULL_QUALITY_WAN_T2V_MODEL_TO_PARAMS = {
    "Wan2.1-T2V-1.3B-Diffusers": WAN_T2V_FULL_QUALITY_PARAMS,
}

WAN_T2V_TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
]


@pytest.mark.parametrize("prompt", WAN_T2V_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN", "TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(WAN_T2V_MODEL_TO_PARAMS.keys()))
def test_wan_t2v_inference_similarity(
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
        default_params_map=WAN_T2V_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_WAN_T2V_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.93,
    )
