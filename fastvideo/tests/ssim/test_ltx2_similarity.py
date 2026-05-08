# SPDX-License-Identifier: Apache-2.0
"""Latent-slice regression test for LTX-2 distilled text-to-video.

Pixel-space SSIM is not a useful signal for this model: 4 distilled
steps + bf16 attention + tiled VAE decode produce outputs that pass
visual QA but occupy a very wide region in pixel space.

Inspired by diffusers' slice-vs-full regression philosophy — see
``diffusers/tests/pipelines/ltx2/test_ltx2.py`` (compares pixel slices
via ``torch.allclose(..., atol=1e-4)``) and
``diffusers/tests/pipelines/cogvideo/test_cogvideox.py`` (full pixel
tensors via ``numpy_cosine_similarity_distance(...) < 1e-3``).
Diffusers itself does NOT compare latents; we apply the same "small
signature slice + bounded full-tensor distance" idea to the **pre-VAE
latent** because distilled few-step pipelines amplify per-step bf16
noise enough that VAE-decoded comparisons are unreliable.

Parameters are kept identical to the original SSIM run so that
reference artefacts generated on Modal L40S remain bit-compatible with
production inference.
"""

import os

import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
)
from fastvideo.tests.ssim.latent_similarity_utils import (
    run_text_to_latent_similarity_test,
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

# Tolerances chosen on top of diffusers' ``1e-3`` defaults. LTX-2 distilled
# amplifies per-step numerical noise, and FastVideo's CI pool spans L40S /
# A40 / H100 so cross-architecture bf16 drift must be absorbed. Values can
# be tightened after an initial stable window of reference refreshes.
SLICE_COSINE_DISTANCE_THRESHOLD = 5e-3
FULL_COSINE_DISTANCE_THRESHOLD = 1e-2


@pytest.mark.parametrize("prompt", LTX2_DISTILLED_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(LTX2_DISTILLED_MODEL_TO_PARAMS.keys()))
def test_ltx2_distilled_inference_similarity(
    prompt: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    run_text_to_latent_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=LTX2_DISTILLED_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_LTX2_DISTILLED_MODEL_TO_PARAMS,
        slice_cosine_threshold=SLICE_COSINE_DISTANCE_THRESHOLD,
        full_cosine_threshold=FULL_COSINE_DISTANCE_THRESHOLD,
    )
