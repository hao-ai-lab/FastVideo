# SPDX-License-Identifier: Apache-2.0
"""Exact clean-frame VAE encoder parity for LingBot-Video TI2V."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLWan as OfficialAutoencoderKLWan
from PIL import Image
from safetensors.torch import load_file

from fastvideo.configs.models.vaes.wanvae import WanVAEConfig
from fastvideo.models.vaes.wanvae import AutoencoderKLWan as FastVideoAutoencoderKLWan
from fastvideo.pipelines.basic.lingbot_video.stages import _preprocess_condition_image
from tests.local_tests.lingbot_video.hf_assets import FASTVIDEO_DENSE, OFFICIAL_DENSE, download_components


def _load_native(checkpoint: Path, device: torch.device) -> FastVideoAutoencoderKLWan:
    """Strict-load the released full VAE into FastVideo's native implementation."""
    vae_dir = checkpoint / "vae"
    raw_config = json.loads((vae_dir / "config.json").read_text())
    config = WanVAEConfig()
    config.update_model_arch({key: value for key, value in raw_config.items() if not key.startswith("_")})
    config.load_encoder = True
    config.load_decoder = True
    model = FastVideoAutoencoderKLWan(config)
    model.load_state_dict(load_file(vae_dir / "diffusion_pytorch_model.safetensors"), strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan VAE parity requires CUDA")
def test_lingbot_video_ti2v_vae_encoder_exact_parity() -> None:
    """Require equal posterior parameters and samples for one clean frame."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on a scheduled GPU")
    device = torch.device("cuda")
    local_official = os.environ.get("LINGBOT_VIDEO_OFFICIAL_CHECKPOINT")
    official_checkpoint = Path(local_official) if local_official else download_components(OFFICIAL_DENSE, "vae")
    local_converted = os.environ.get("LINGBOT_VIDEO_TI2V_CHECKPOINT")
    converted_checkpoint = Path(local_converted) if local_converted else download_components(FASTVIDEO_DENSE, "vae")
    y, x = np.mgrid[0:941, 0:1672]
    image = Image.fromarray(
        np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8),
        mode="RGB",
    )
    pixels = _preprocess_condition_image(image, 480, 832).to(device)
    normalized = (pixels - 0.5) / 0.5
    official = OfficialAutoencoderKLWan.from_pretrained(
        official_checkpoint / "vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device).eval()
    native = _load_native(converted_checkpoint, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = official.encode(normalized).latent_dist
        actual = native.encode(normalized)
    expected_generator = torch.Generator(device=device).manual_seed(42)
    actual_generator = torch.Generator(device=device).manual_seed(42)
    expected_sample = expected.sample(expected_generator)
    actual_sample = actual.sample(actual_generator)
    for name, expected_tensor, actual_tensor in (
        ("mean", expected.mean, actual.mean),
        ("logvar", expected.logvar, actual.logvar),
        ("sample", expected_sample, actual_sample),
    ):
        difference = (actual_tensor.float() - expected_tensor.float()).abs()
        print(
            f"{name}: equal={torch.equal(actual_tensor, expected_tensor)} "
            f"differing={torch.count_nonzero(actual_tensor != expected_tensor).item()} "
            f"max_abs={difference.max().item():.8f} mean_abs={difference.mean().item():.8f}"
        )
        assert torch.equal(actual_tensor, expected_tensor)
