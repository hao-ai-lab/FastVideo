# SPDX-License-Identifier: Apache-2.0
"""SSIM-based similarity test for daVinci-MagiHuman base text-to-AV.

Reference videos for this test are seeded separately via the
`.agents/skills/seed-ssim-references/` skill on Modal L40S and uploaded
to `FastVideo/ssim-reference-videos`. Until refs exist, the first run
will fail downloading; run the seed skill once and commit the URLs.

Resolution + steps kept small enough for a CI budget; the full-quality
variant falls back to the registered preset defaults.
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

REQUIRED_GPUS = 1

device_reference_folder = resolve_inference_device_reference_folder(logger)

# NOTE: update `model_path` once the converted repo is published. Until
# then, point at your local converted_weights directory or skip via env.
_MAGI_HUMAN_MODEL_PATH = os.getenv(
    "MAGI_HUMAN_MODEL_PATH",
    "FastVideo/MagiHuman-Base-Diffusers",
)

MAGI_HUMAN_BASE_PARAMS = {
    "num_gpus": 1,
    "model_path": _MAGI_HUMAN_MODEL_PATH,
    "height": 256,
    "width": 448,
    "num_frames": 26,                # seconds=1 at fps=25 + 1
    "num_inference_steps": 4,        # CI budget; full preset uses 32
    "guidance_scale": 5.0,
    "seed": 42,
    "sp_size": 1,
    "tp_size": 1,
    "fps": 25,
}

try:
    _MAGI_HUMAN_FULL_DEFAULTS = SamplingParam.from_pretrained(_MAGI_HUMAN_MODEL_PATH)
    MAGI_HUMAN_FULL_PARAMS = {
        "num_gpus": MAGI_HUMAN_BASE_PARAMS["num_gpus"],
        "model_path": MAGI_HUMAN_BASE_PARAMS["model_path"],
        "height": _MAGI_HUMAN_FULL_DEFAULTS.height,
        "width": _MAGI_HUMAN_FULL_DEFAULTS.width,
        "num_frames": _MAGI_HUMAN_FULL_DEFAULTS.num_frames,
        "num_inference_steps": _MAGI_HUMAN_FULL_DEFAULTS.num_inference_steps,
        "guidance_scale": _MAGI_HUMAN_FULL_DEFAULTS.guidance_scale,
        "seed": _MAGI_HUMAN_FULL_DEFAULTS.seed,
        "sp_size": MAGI_HUMAN_BASE_PARAMS["sp_size"],
        "tp_size": MAGI_HUMAN_BASE_PARAMS["tp_size"],
        "fps": _MAGI_HUMAN_FULL_DEFAULTS.fps,
    }
except Exception:
    # Model not registered / accessible on this machine — fall back to the
    # quick params as the full-quality map too; the test will skip anyway
    # when the model path is unavailable.
    MAGI_HUMAN_FULL_PARAMS = MAGI_HUMAN_BASE_PARAMS


MAGI_HUMAN_MODEL_TO_PARAMS = {
    "MagiHuman-Base-Diffusers": MAGI_HUMAN_BASE_PARAMS,
}
FULL_QUALITY_MAGI_HUMAN_MODEL_TO_PARAMS = {
    "MagiHuman-Base-Diffusers": MAGI_HUMAN_FULL_PARAMS,
}

MAGI_HUMAN_TEST_PROMPTS = [
    "A person sitting by a window, softly lit by afternoon sun, waving at "
    "the camera with a gentle smile.",
]


@pytest.mark.parametrize("prompt", MAGI_HUMAN_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(MAGI_HUMAN_MODEL_TO_PARAMS.keys()))
def test_magi_human_base_inference_similarity(
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
        default_params_map=MAGI_HUMAN_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_MAGI_HUMAN_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.60,
    )
