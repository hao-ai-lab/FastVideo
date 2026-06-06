# SPDX-License-Identifier: Apache-2.0
"""Latent-slice regression tests for Flux2 text-to-image variants.

Flux2 currently has local parity coverage against the official/reference
pipeline, but CI needs a small deterministic regression gate for seeded HF
artefacts.  Pixel-space comparisons are unnecessarily brittle for this first
slot, so the test follows the latent helper pattern used by LTX-2: generate a
single-image latent with the production recipe, persist the generated latent,
and compare a stable latent signature plus the full tensor against the device
reference.

The default and full-quality parameter maps intentionally carry the same
recipe values for now.  The ``--ssim-full-quality`` flag still switches the
reference tier through ``conftest.py``; separate full-quality recipes can be
introduced after the initial Flux2 references have a stable CI window.
"""

from __future__ import annotations

import os

import pytest
import torch

from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
)
from fastvideo.tests.ssim.latent_similarity_utils import (
    run_text_to_latent_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_inference_device_reference_folder(logger)

FLUX2_DEV_MODEL_ID = "black-forest-labs/FLUX.2-dev"
FLUX2_KLEIN_4B_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_9B_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

FLUX2_MODEL_TO_PARAMS: dict[str, dict[str, object]] = {
    FLUX2_DEV_MODEL_ID: {
        "num_gpus": 1,
        "model_path": FLUX2_DEV_MODEL_ID,
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "seed": 0,
        "sp_size": 1,
        "tp_size": 1,
        "fps": 1,
    },
    FLUX2_KLEIN_4B_MODEL_ID: {
        "num_gpus": 1,
        "model_path": FLUX2_KLEIN_4B_MODEL_ID,
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": 0,
        "sp_size": 1,
        "tp_size": 1,
        "fps": 1,
    },
    FLUX2_KLEIN_9B_MODEL_ID: {
        "num_gpus": 1,
        "model_path": FLUX2_KLEIN_9B_MODEL_ID,
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": 0,
        "sp_size": 1,
        "tp_size": 1,
        "fps": 1,
    },
}

FLUX2_FULL_QUALITY_MODEL_TO_PARAMS: dict[str, dict[str, object]] = {
    model_id: dict(params)
    for model_id, params in FLUX2_MODEL_TO_PARAMS.items()
}

TEST_PROMPTS: dict[str, str] = {
    FLUX2_DEV_MODEL_ID: "a photo of a banana on a wooden table, studio lighting",
    FLUX2_KLEIN_4B_MODEL_ID: "a brushed steel espresso machine on a marble counter, morning window light",
    FLUX2_KLEIN_9B_MODEL_ID: "a brushed steel espresso machine on a marble counter, morning window light",
}

SLICE_COSINE_THRESHOLD = 0.96
FULL_COSINE_THRESHOLD = 0.99


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Flux2 SSIM test requires CUDA",
)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(FLUX2_MODEL_TO_PARAMS.keys()))
def test_flux2_similarity(
    attention_backend_name: str,
    model_id: str,
) -> None:
    _ = run_text_to_latent_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=TEST_PROMPTS[model_id],
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=FLUX2_MODEL_TO_PARAMS,
        full_quality_params_map=FLUX2_FULL_QUALITY_MODEL_TO_PARAMS,
        slice_cosine_threshold=SLICE_COSINE_THRESHOLD,
        full_cosine_threshold=FULL_COSINE_THRESHOLD,
        init_kwargs_override={
            "workload_type": "t2i",
            "use_fsdp_inference": False,
            "dit_cpu_offload": False,
            "vae_cpu_offload": True,
            "text_encoder_cpu_offload": True,
            "pin_cpu_memory": False,
        },
    )
