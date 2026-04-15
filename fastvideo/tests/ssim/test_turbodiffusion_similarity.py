# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from fastvideo.configs.sample.base import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    DEVICE_MAPPINGS,
    resolve_inference_device_reference_folder,
    run_image_to_video_similarity_test,
    run_text_to_video_similarity_test,
)
from fastvideo.tests.ssim.reference_utils import (
    get_cuda_device_name,
    resolve_device_reference_folder,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 4

device_name = get_cuda_device_name()
device_reference_folder = resolve_device_reference_folder(
    DEVICE_MAPPINGS,
    device_name=device_name,
)
if device_reference_folder is None:
    raise ValueError(f"Unsupported device for ssim tests: {device_name}")

TURBODIFFUSION_PARAMS = {
    "num_gpus": 4,
    "model_path": "loayrashid/TurboWan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 42,
    "sp_size": 4,
    "tp_size": 1,
    "fps": 24,
}
_TURBODIFFUSION_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    TURBODIFFUSION_PARAMS["model_path"])
TURBODIFFUSION_FULL_QUALITY_PARAMS = {
    "num_gpus": TURBODIFFUSION_PARAMS["num_gpus"],
    "model_path": TURBODIFFUSION_PARAMS["model_path"],
    "height": _TURBODIFFUSION_FULL_QUALITY_DEFAULTS.height,
    "width": _TURBODIFFUSION_FULL_QUALITY_DEFAULTS.width,
    "num_frames": TURBODIFFUSION_PARAMS["num_frames"],
    "num_inference_steps": (_TURBODIFFUSION_FULL_QUALITY_DEFAULTS.num_inference_steps),
    "guidance_scale": (_TURBODIFFUSION_FULL_QUALITY_DEFAULTS.guidance_scale),
    "seed": _TURBODIFFUSION_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": TURBODIFFUSION_PARAMS["sp_size"],
    "tp_size": TURBODIFFUSION_PARAMS["tp_size"],
    "fps": _TURBODIFFUSION_FULL_QUALITY_DEFAULTS.fps,
}

TURBODIFFUSION_MODEL_TO_PARAMS = {
    "TurboWan2.1-T2V-1.3B-Diffusers": TURBODIFFUSION_PARAMS,
}
FULL_QUALITY_TURBODIFFUSION_MODEL_TO_PARAMS = {
    "TurboWan2.1-T2V-1.3B-Diffusers": (TURBODIFFUSION_FULL_QUALITY_PARAMS),
}

TURBODIFFUSION_TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting "
    "with the energetic background of a bustling street food market. "
    "The scene captures a mix of humor and authenticity. Mid-shot "
    "framing, vibrant lighting.",
]


@pytest.mark.parametrize("prompt", TURBODIFFUSION_TEST_PROMPTS)
@pytest.mark.parametrize(
    "model_id",
    list(TURBODIFFUSION_MODEL_TO_PARAMS.keys()),
)
def test_turbodiffusion_inference_similarity(
    prompt: str,
    model_id: str,
) -> None:
    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name="SLA_ATTN",
        model_id=model_id,
        default_params_map=TURBODIFFUSION_MODEL_TO_PARAMS,
        full_quality_params_map=(FULL_QUALITY_TURBODIFFUSION_MODEL_TO_PARAMS),
        min_acceptable_ssim=0.95,
        init_kwargs_override={
            "override_pipeline_cls_name": "TurboDiffusionPipeline",
        },
    )


TURBODIFFUSION_I2V_PARAMS = {
    "num_gpus": 4,
    "model_path": "loayrashid/TurboWan2.2-I2V-A14B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 42,
    "sp_size": 4,
    "tp_size": 1,
    "fps": 24,
}
_TURBODIFFUSION_I2V_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    TURBODIFFUSION_I2V_PARAMS["model_path"])
TURBODIFFUSION_I2V_FULL_QUALITY_PARAMS = {
    "num_gpus": TURBODIFFUSION_I2V_PARAMS["num_gpus"],
    "model_path": TURBODIFFUSION_I2V_PARAMS["model_path"],
    "height": _TURBODIFFUSION_I2V_FULL_QUALITY_DEFAULTS.height,
    "width": _TURBODIFFUSION_I2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": TURBODIFFUSION_I2V_PARAMS["num_frames"],
    "num_inference_steps": (_TURBODIFFUSION_I2V_FULL_QUALITY_DEFAULTS.num_inference_steps),
    "guidance_scale": (_TURBODIFFUSION_I2V_FULL_QUALITY_DEFAULTS.guidance_scale),
    "seed": _TURBODIFFUSION_I2V_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": TURBODIFFUSION_I2V_PARAMS["sp_size"],
    "tp_size": TURBODIFFUSION_I2V_PARAMS["tp_size"],
    "fps": _TURBODIFFUSION_I2V_FULL_QUALITY_DEFAULTS.fps,
}

TURBODIFFUSION_I2V_MODEL_TO_PARAMS = {
    "TurboWan2.2-I2V-A14B-Diffusers": TURBODIFFUSION_I2V_PARAMS,
}
FULL_QUALITY_TURBODIFFUSION_I2V_MODEL_TO_PARAMS = {
    "TurboWan2.2-I2V-A14B-Diffusers": (TURBODIFFUSION_I2V_FULL_QUALITY_PARAMS),
}

TURBODIFFUSION_I2V_TEST_PROMPTS = [
    "An astronaut hatching from an egg, on the surface of the moon, "
    "the darkness and depth of space realised in the background. "
    "High quality, ultrarealistic detail and breath-taking "
    "movie-like camera shot.",
]

TURBODIFFUSION_I2V_IMAGE_PATHS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
]


@pytest.mark.parametrize("prompt", TURBODIFFUSION_I2V_TEST_PROMPTS)
@pytest.mark.parametrize(
    "model_id",
    list(TURBODIFFUSION_I2V_MODEL_TO_PARAMS.keys()),
)
def test_turbodiffusion_i2v_inference_similarity(
    prompt: str,
    model_id: str,
) -> None:
    image_path = TURBODIFFUSION_I2V_IMAGE_PATHS[TURBODIFFUSION_I2V_TEST_PROMPTS.index(prompt)]
    run_image_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        image_path=image_path,
        attention_backend_name="SLA_ATTN",
        model_id=model_id,
        default_params_map=TURBODIFFUSION_I2V_MODEL_TO_PARAMS,
        full_quality_params_map=(FULL_QUALITY_TURBODIFFUSION_I2V_MODEL_TO_PARAMS),
        min_acceptable_ssim=0.95,
        init_kwargs_override={
            "override_pipeline_cls_name": ("TurboDiffusionI2VPipeline"),
        },
    )
