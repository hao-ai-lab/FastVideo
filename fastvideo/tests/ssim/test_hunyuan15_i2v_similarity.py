# SPDX-License-Identifier: Apache-2.0
"""End-to-end SSIM regression test for HunyuanVideo 1.5 image-to-video.

The CPU stage tests fake the SigLIP tower and the VAE, so the two things that
can only fail against the real checkpoint are unverified there: the loader
wiring (the i2v checkpoints ship ``image_encoder``/``feature_extractor``) and
the numerics (scaling factor, reference-image resize, mask placement). This
test runs the real checkpoint with an image and a prompt and compares the clip
against a device-specific reference, which is what catches a regression that
silently drops the reference image.
"""
import os

import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
    run_image_to_video_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 2

device_reference_folder = resolve_inference_device_reference_folder(logger)

HUNYUAN15_I2V_PARAMS = {
    "num_gpus": 2,
    "model_path": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
    "height": 480,
    "width": 848,
    "num_frames": 45,
    "num_inference_steps": 6,
    "guidance_scale": 1.0,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 1,
    "fps": 24,
}
_HUNYUAN15_I2V_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(HUNYUAN15_I2V_PARAMS["model_path"])
HUNYUAN15_I2V_FULL_QUALITY_PARAMS = {
    "num_gpus": HUNYUAN15_I2V_PARAMS["num_gpus"],
    "model_path": HUNYUAN15_I2V_PARAMS["model_path"],
    "height": _HUNYUAN15_I2V_FULL_QUALITY_DEFAULTS.height,
    "width": _HUNYUAN15_I2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": _HUNYUAN15_I2V_FULL_QUALITY_DEFAULTS.num_frames,
    "num_inference_steps": _HUNYUAN15_I2V_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _HUNYUAN15_I2V_FULL_QUALITY_DEFAULTS.guidance_scale,
    "seed": _HUNYUAN15_I2V_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": HUNYUAN15_I2V_PARAMS["sp_size"],
    "tp_size": HUNYUAN15_I2V_PARAMS["tp_size"],
    "fps": _HUNYUAN15_I2V_FULL_QUALITY_DEFAULTS.fps,
}

HUNYUAN15_I2V_MODEL_TO_PARAMS = {
    "HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled": HUNYUAN15_I2V_PARAMS,
}
FULL_QUALITY_HUNYUAN15_I2V_MODEL_TO_PARAMS = {
    "HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled": HUNYUAN15_I2V_FULL_QUALITY_PARAMS,
}

HUNYUAN15_I2V_TEST_CASES = [
    (
        "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
    ),
]


@pytest.mark.parametrize(("prompt", "image_path"), HUNYUAN15_I2V_TEST_CASES)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(HUNYUAN15_I2V_MODEL_TO_PARAMS.keys()))
def test_hunyuan15_i2v_inference_similarity(
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
        default_params_map=HUNYUAN15_I2V_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_HUNYUAN15_I2V_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.95,
    )
