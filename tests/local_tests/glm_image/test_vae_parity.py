# SPDX-License-Identifier: Apache-2.0
"""Component parity scaffold for GLM-Image VAE (AutoencoderKL).

Compares the FastVideo-native AutoencoderKL port against diffusers'
`AutoencoderKL` loaded from `zai-org/GLM-Image/vae`. The published checkpoint
uses `block_out_channels=[128, 512, 1024, 1024]`, `latent_channels=16`, and
per-channel `latents_mean[16]` + `latents_std[16]` normalization (not a scalar
`scaling_factor`).

Skips cleanly until both the FastVideo class and the VAE shard exist.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")
os.environ.setdefault("DISABLE_SP", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY = "glm_image"
LOCAL_WEIGHTS_DIR = Path(
    os.getenv("GLM_IMAGE_LOCAL_WEIGHTS_DIR",
              REPO_ROOT / "official_weights" / FAMILY))
VAE_DIR = LOCAL_WEIGHTS_DIR / "vae"


def _has_weights() -> bool:
    return (VAE_DIR / "diffusion_pytorch_model.safetensors").exists()


pytestmark = pytest.mark.skipif(
    not _has_weights(),
    reason=f"GLM-Image VAE weights not found at {VAE_DIR}.",
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GLM-Image VAE parity.")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def vae_config():
    with open(VAE_DIR / "config.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def official_vae(device):
    diffusers = pytest.importorskip("diffusers")
    vae = diffusers.AutoencoderKL.from_pretrained(
        str(VAE_DIR), torch_dtype=torch.float32).to(device).eval()
    return vae


@pytest.fixture(scope="module")
def fastvideo_vae(device):
    pytest.importorskip("fastvideo")
    try:
        from fastvideo.configs.models.vaes.glm_image import GlmImageVAEConfig
        from fastvideo.models.vaes.autoencoder_kl import AutoencoderKL
    except ImportError as e:
        pytest.skip(f"FastVideo native AutoencoderKL not yet ported: {e}")
    cfg = GlmImageVAEConfig()
    vae = AutoencoderKL(cfg)
    safetensors = pytest.importorskip("safetensors.torch")
    sd = safetensors.load_file(
        str(VAE_DIR / "diffusion_pytorch_model.safetensors"))
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    assert not missing, f"FastVideo VAE missing keys: {missing[:10]}"
    assert not unexpected, f"FastVideo VAE unexpected keys: {unexpected[:10]}"
    return vae.to(device, dtype=torch.float32).eval()


def test_vae_config_matches_real_checkpoint(vae_config):
    assert vae_config["block_out_channels"] == [128, 512, 1024, 1024]
    assert vae_config["latent_channels"] == 16
    assert "latents_mean" in vae_config and len(
        vae_config["latents_mean"]) == 16
    assert "latents_std" in vae_config and len(vae_config["latents_std"]) == 16


def test_vae_decode_matches_diffusers(official_vae, fastvideo_vae, device):
    torch.manual_seed(0)
    latents = torch.randn(1, 16, 32, 32, device=device, dtype=torch.float32)
    with torch.no_grad():
        official_out = official_vae.decode(latents, return_dict=False)[0]
        fv_out = fastvideo_vae.decode(latents, return_dict=False)[0]
    torch.testing.assert_close(fv_out, official_out, atol=1e-3, rtol=1e-3)


def test_vae_encode_matches_diffusers(official_vae, fastvideo_vae, device):
    torch.manual_seed(0)
    pixels = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
    with torch.no_grad():
        official_z = official_vae.encode(pixels).latent_dist.mode()
        fv_z = fastvideo_vae.encode(pixels).latent_dist.mode()
    torch.testing.assert_close(fv_z, official_z, atol=1e-3, rtol=1e-3)
