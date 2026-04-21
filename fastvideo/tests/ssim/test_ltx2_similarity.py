# SPDX-License-Identifier: Apache-2.0
"""SSIM-based similarity test for LTX-2 distilled text-to-video.

Parameters derived from examples/inference/basic/basic_ltx2_distilled.py,
with resolution + num_inference_steps reduced to keep GPU CI runtime
bounded. Full-quality variant (via ``--ssim-full-quality``) falls back
to the ``ltx2_distilled`` preset defaults.
"""
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

LTX2_DISTILLED_PARAMS = {
    "num_gpus": 2,
    "model_path": "FastVideo/LTX2-Distilled-Diffusers",
    "height": 512,
    "width": 768,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 10,
    "sp_size": 2,
    "tp_size": 1,
    "fps": 24,
    "ltx2_vae_tiling": True,
}
_LTX2_DISTILLED_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    LTX2_DISTILLED_PARAMS["model_path"])
LTX2_DISTILLED_FULL_QUALITY_PARAMS = {
    "num_gpus": LTX2_DISTILLED_PARAMS["num_gpus"],
    "model_path": LTX2_DISTILLED_PARAMS["model_path"],
    "height": _LTX2_DISTILLED_FULL_QUALITY_DEFAULTS.height,
    "width": _LTX2_DISTILLED_FULL_QUALITY_DEFAULTS.width,
    "num_frames": _LTX2_DISTILLED_FULL_QUALITY_DEFAULTS.num_frames,
    "num_inference_steps":
        _LTX2_DISTILLED_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _LTX2_DISTILLED_FULL_QUALITY_DEFAULTS.guidance_scale,
    "seed": _LTX2_DISTILLED_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": LTX2_DISTILLED_PARAMS["sp_size"],
    "tp_size": LTX2_DISTILLED_PARAMS["tp_size"],
    "fps": _LTX2_DISTILLED_FULL_QUALITY_DEFAULTS.fps,
    "ltx2_vae_tiling": LTX2_DISTILLED_PARAMS["ltx2_vae_tiling"],
}

LTX2_DISTILLED_MODEL_TO_PARAMS = {
    "LTX2-Distilled-Diffusers": LTX2_DISTILLED_PARAMS,
}
FULL_QUALITY_LTX2_DISTILLED_MODEL_TO_PARAMS = {
    "LTX2-Distilled-Diffusers": LTX2_DISTILLED_FULL_QUALITY_PARAMS,
}

LTX2_DISTILLED_TEST_PROMPTS = [
    "A warm sunny backyard. The camera starts in a tight cinematic "
    "close-up of a woman and a man in their 30s, facing each other with "
    "serious expressions. The camera slowly pans right, revealing a "
    "grandfather in the garden wearing enormous butterfly wings, waving "
    "his arms in the air like he's trying to take off. The tone is "
    "deadpan, absurd, and quietly tragic.",
]


@pytest.mark.parametrize("prompt", LTX2_DISTILLED_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(LTX2_DISTILLED_MODEL_TO_PARAMS.keys()))
def test_ltx2_distilled_inference_similarity(
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
        default_params_map=LTX2_DISTILLED_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_LTX2_DISTILLED_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.98,
    )
