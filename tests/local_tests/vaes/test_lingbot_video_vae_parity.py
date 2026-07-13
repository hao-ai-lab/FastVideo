# SPDX-License-Identifier: Apache-2.0
"""LingBot-Video Wan VAE parity against the released Diffusers component.

Coverage scope: implementation_subcomponent. The released Dense and MoE roots
share the same VAE checkpoint. The test uses the exact production architecture
config and compares deterministic encode-mode and decode tensors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLWan as OfficialAutoencoderKLWan
from safetensors.torch import load_file
from torch.testing import assert_close

from fastvideo.configs.models.vaes.wanvae import WanVAEConfig
from fastvideo.models.vaes.wanvae import AutoencoderKLWan as FastVideoAutoencoderKLWan
from tests.local_tests.lingbot_video.hf_assets import (
    FASTVIDEO_DENSE,
    OFFICIAL_DENSE,
    download_components,
)

PARITY_SCOPE = "implementation_subcomponent"


def _load_fastvideo_vae(vae_dir: Path, device: torch.device) -> FastVideoAutoencoderKLWan:
    """Strict-load the released VAE weights into FastVideo's native class."""
    config_path = vae_dir / "config.json"
    weights_path = vae_dir / "diffusion_pytorch_model.safetensors"
    raw_config = json.loads(config_path.read_text())
    config = WanVAEConfig()
    config.update_model_arch({key: value for key, value in raw_config.items() if not key.startswith("_")})
    config.load_encoder = True
    config.load_decoder = True
    model = FastVideoAutoencoderKLWan(config)
    model.load_state_dict(load_file(weights_path), strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LingBot-Video VAE parity requires CUDA.")
def test_lingbot_video_vae_decode_matches_official() -> None:
    """Decode the same normalized latent through both VAE implementations."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("Set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on a scheduled GPU node.")
    device = torch.device("cuda:0")
    official_root = download_components(OFFICIAL_DENSE, "vae")
    fastvideo_root = download_components(FASTVIDEO_DENSE, "vae")
    official = OfficialAutoencoderKLWan.from_pretrained(
        official_root / "vae", torch_dtype=torch.float32
    ).to(device).eval()
    fastvideo = _load_fastvideo_vae(fastvideo_root / "vae", device)
    generator = torch.Generator(device=device).manual_seed(42)
    latents = torch.randn((1, 16, 3, 8, 8), generator=generator, device=device, dtype=torch.float32)
    mean = torch.tensor(official.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
    std = torch.tensor(official.config.latents_std, device=device).view(1, 16, 1, 1, 1)
    decoder_latents = latents * std + mean

    with torch.inference_mode():
        official_output = official.decode(decoder_latents, return_dict=False)[0].float()
        fastvideo_output = fastvideo.decode(decoder_latents).float()
    assert_close(fastvideo_output, official_output, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LingBot-Video VAE parity requires CUDA.")
def test_lingbot_video_vae_encode_mode_matches_official() -> None:
    """Compare deterministic posterior modes before pipeline normalization."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("Set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on a scheduled GPU node.")
    device = torch.device("cuda:0")
    official_root = download_components(OFFICIAL_DENSE, "vae")
    fastvideo_root = download_components(FASTVIDEO_DENSE, "vae")
    official = OfficialAutoencoderKLWan.from_pretrained(
        official_root / "vae", torch_dtype=torch.float32
    ).to(device).eval()
    fastvideo = _load_fastvideo_vae(fastvideo_root / "vae", device)
    generator = torch.Generator(device=device).manual_seed(43)
    video = torch.randn((1, 3, 9, 64, 64), generator=generator, device=device, dtype=torch.float32)

    with torch.inference_mode():
        official_mode = official.encode(video).latent_dist.mode().float()
        fastvideo_mode = fastvideo.encode(video).mode().float()
    assert_close(fastvideo_mode, official_mode, atol=5e-2, rtol=5e-2)
