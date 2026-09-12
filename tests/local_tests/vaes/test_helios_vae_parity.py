# SPDX-License-Identifier: Apache-2.0
"""Helios exact-checkpoint Wan VAE reuse parity.

Coverage scope: implementation_subcomponent. This is intentionally a real
weight test: matching config fields alone is insufficient evidence that the
existing FastVideo Wan VAE can safely serve Helios.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLWan as OfficialAutoencoderKLWan
from safetensors.torch import load_file
from torch.testing import assert_close

from fastvideo.configs.models.vaes.wanvae import WanVAEArchConfig, WanVAEConfig
from fastvideo.models.vaes.wanvae import AutoencoderKLWan

REPO_ROOT = Path(__file__).resolve().parents[3]
VAE_DIR = REPO_ROOT / "official_weights" / "helios" / "vae"
PARITY_SCOPE = "implementation_subcomponent"


def _require_weights() -> Path:
    weight_path = VAE_DIR / "diffusion_pytorch_model.safetensors"
    if not weight_path.exists():
        pytest.skip(f"Helios VAE weights missing: {weight_path}")
    return weight_path


def _fastvideo_vae(device: torch.device) -> AutoencoderKLWan:
    config = WanVAEConfig(arch_config=WanVAEArchConfig())
    config.load_encoder = True
    config.load_decoder = True
    model = AutoencoderKLWan(config).to(device=device, dtype=torch.float32)
    incompatible = model.load_state_dict(load_file(_require_weights()), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    return model.eval()


def test_helios_vae_config_matches_native_wan_candidate() -> None:
    config = WanVAEArchConfig()
    assert config.base_dim == 96
    assert config.z_dim == 16
    assert config.dim_mult == (1, 2, 4, 4)
    assert config.temperal_downsample == (False, True, True)
    assert config.scale_factor_temporal == 4
    assert config.scale_factor_spatial == 8
    assert len(config.latents_mean) == len(config.latents_std) == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VAE parity.")
def test_helios_vae_decode_parity() -> None:
    device = torch.device("cuda:0")
    official = (OfficialAutoencoderKLWan.from_pretrained(str(VAE_DIR), local_files_only=True,
                                                         torch_dtype=torch.float32).to(device).eval())
    fastvideo = _fastvideo_vae(device)

    generator = torch.Generator(device=device).manual_seed(4242)
    latents = torch.randn(1, 16, 2, 8, 8, device=device, generator=generator)
    with torch.inference_mode():
        official_output = official.decode(latents, return_dict=False)[0].float().cpu()
        fastvideo_output = fastvideo.decode(latents).float().cpu()

    assert official_output.shape == fastvideo_output.shape
    diff = (official_output - fastvideo_output).abs()
    print(f"VAE decode diff_max={diff.max().item():.8f} diff_mean={diff.mean().item():.8f}")
    assert_close(fastvideo_output, official_output, atol=1e-4, rtol=1e-4)
