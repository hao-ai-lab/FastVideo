# SPDX-License-Identifier: Apache-2.0
"""SSIM regression test for Kandinsky-6.0 Pro text+image-to-video (TI2VA).

Runs a reduced-size generation (256x384, 21 frames) with a fixed seed and compares against a
device-specific reference video via MS-SSIM. The preset default is 512x768 x 121 frames; the reduced
size keeps CI cheap while the patch (1, 2, 2) and VAE temporal-4 / spatial-8 constraints stay satisfied
(dims divisible by 16, num_frames % 4 == 1) -- same constraints as Kandinsky-5's DiT, which Kandinsky6
extends (see fastvideo/models/dits/kandinsky6.py's module docstring).

No reference video exists yet for this test; generating and uploading one needs a GPU run against the
checkpoint plus the fastvideo/tests/ssim/reference_videos_cli.py upload flow documented in
fastvideo/tests/ssim/AGENTS.md. Until then this fails with a clear "run reference_videos_cli.py download" message (or xfails, in
bootstrap mode) the same way every other SSIM test does when its reference is missing -- see
_assert_similarity in inference_similarity_utils.py.
"""
import os

import pytest

from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.kandinsky6.presets import KANDINSKY6_TI2VA
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
    run_text_to_video_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_inference_device_reference_folder(logger)

_PRESET_DEFAULTS = KANDINSKY6_TI2VA.defaults

KANDINSKY6_TI2VA_PARAMS = {
    "num_gpus": 1,
    "model_path": "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers",
    "height": 256,
    "width": 384,
    "num_frames": 21,
    "num_inference_steps": _PRESET_DEFAULTS["num_inference_steps"],
    "guidance_scale": _PRESET_DEFAULTS["guidance_scale"],
    "seed": 1024,
    "sp_size": 1,
    "tp_size": 1,
    "fps": _PRESET_DEFAULTS["fps"],
    "neg_prompt": _PRESET_DEFAULTS["negative_prompt"],
}

KANDINSKY6_TI2VA_FULL_QUALITY_PARAMS = {
    **KANDINSKY6_TI2VA_PARAMS,
    "height": _PRESET_DEFAULTS["height"],
    "width": _PRESET_DEFAULTS["width"],
    # num_frames stays at the inherited reduced 21 even at full quality (preset default: 121).
}

KANDINSKY6_TI2VA_MODEL_TO_PARAMS = {
    "Kandinsky-6.0-Pro-5s-Diffusers": KANDINSKY6_TI2VA_PARAMS,
}
FULL_QUALITY_KANDINSKY6_TI2VA_MODEL_TO_PARAMS = {
    "Kandinsky-6.0-Pro-5s-Diffusers": KANDINSKY6_TI2VA_FULL_QUALITY_PARAMS,
}

KANDINSKY6_TI2VA_TEST_PROMPTS = [
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its "
    "eyes wide with interest. The playful yet serene atmosphere is complemented "
    "by soft natural light filtering through the petals. Mid-shot, warm and "
    "cheerful tones.",
]


@pytest.mark.parametrize("prompt", KANDINSKY6_TI2VA_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(KANDINSKY6_TI2VA_MODEL_TO_PARAMS.keys()))
def test_kandinsky6_ti2va_inference_similarity(
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
        default_params_map=KANDINSKY6_TI2VA_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_KANDINSKY6_TI2VA_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.93,
        # Kandinsky6 shares Kandinsky5's inference-path conventions (see
        # examples/inference/basic/basic_kandinsky6_ti2va.py): no FSDP inference, text encoder offloaded
        # to CPU between encodes.
        init_kwargs_override={
            "use_fsdp_inference": False,
            "text_encoder_cpu_offload": True,
            "pin_cpu_memory": True,
        },
    )
