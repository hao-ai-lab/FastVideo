# SPDX-License-Identifier: Apache-2.0
"""
Parity test for Z-Image VAE (AutoencoderKL decode path).

This compares FastVideo's AutoencoderKL wrapper against the Z-Image
reference AutoencoderKL implementation using the same local weights.

Usage:
    pytest tests/local_tests/zimage/test_zimage_vae_parity.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file as safetensors_load_file
from torch.testing import assert_close

from fastvideo.configs.models.vaes.autoencoder_kl import AutoencoderKLVAEConfig
from fastvideo.models.vaes.autoencoder_kl import AutoencoderKL as FastVideoAutoencoderKL


REPO_ROOT = Path(__file__).resolve().parents[3]
ZIMAGE_SRC = REPO_ROOT / "Z-Image" / "src"
ZIMAGE_VAE_DIR = REPO_ROOT / "official_weights" / "Z-Image" / "vae"
ZIMAGE_VAE_CFG = ZIMAGE_VAE_DIR / "config.json"
ZIMAGE_VAE_WEIGHTS = ZIMAGE_VAE_DIR / "diffusion_pytorch_model.safetensors"

if str(ZIMAGE_SRC) not in sys.path:
    sys.path.insert(0, str(ZIMAGE_SRC))

try:
    from zimage.autoencoder import AutoencoderKL as ReferenceAutoencoderKL
except Exception as exc:  # pragma: no cover - handled by skip
    ReferenceAutoencoderKL = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _load_cfg() -> dict:
    if not ZIMAGE_VAE_CFG.exists():
        pytest.skip(f"Z-Image VAE config not found: {ZIMAGE_VAE_CFG}")
    with ZIMAGE_VAE_CFG.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    cfg.pop("_name_or_path", None)
    return cfg


def _load_weights() -> dict[str, torch.Tensor]:
    if not ZIMAGE_VAE_WEIGHTS.exists():
        pytest.skip(f"Z-Image VAE weights not found: {ZIMAGE_VAE_WEIGHTS}")
    return safetensors_load_file(str(ZIMAGE_VAE_WEIGHTS), device="cpu")


def _build_reference(cfg: dict, weights: dict[str, torch.Tensor]) -> torch.nn.Module:
    if ReferenceAutoencoderKL is None:
        pytest.skip(f"Cannot import Z-Image reference autoencoder: {_IMPORT_ERROR}")

    ref = ReferenceAutoencoderKL(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        down_block_types=tuple(cfg["down_block_types"]),
        up_block_types=tuple(cfg["up_block_types"]),
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg["layers_per_block"],
        latent_channels=cfg["latent_channels"],
        norm_num_groups=cfg["norm_num_groups"],
        scaling_factor=cfg["scaling_factor"],
        shift_factor=cfg["shift_factor"],
        use_quant_conv=cfg["use_quant_conv"],
        use_post_quant_conv=cfg["use_post_quant_conv"],
        mid_block_add_attention=cfg["mid_block_add_attention"],
    ).eval()
    ref.load_state_dict(weights, strict=True)
    return ref


def _build_fastvideo(cfg: dict, weights: dict[str, torch.Tensor]) -> torch.nn.Module:
    fv_cfg = AutoencoderKLVAEConfig()
    fv_cfg.update_model_arch(cfg)
    fv = FastVideoAutoencoderKL(fv_cfg).eval()
    fv.load_state_dict(weights, strict=True)
    return fv


def test_zimage_vae_decode_parity():
    torch.manual_seed(7)

    cfg = _load_cfg()
    weights = _load_weights()

    ref = _build_reference(cfg, weights)
    fv = _build_fastvideo(cfg, weights)

    # Decode-only parity is the critical path for Z-Image pipeline integration.
    latents = torch.randn(1, cfg["latent_channels"], 8, 8, dtype=torch.float32)

    with torch.no_grad():
        ref_out = ref.decode(latents, return_dict=False)[0].detach().float()
        fv_out = fv.decode(latents, return_dict=False)[0].detach().float()

    assert ref_out.shape == fv_out.shape
    assert_close(ref_out, fv_out, atol=1e-4, rtol=1e-4)
