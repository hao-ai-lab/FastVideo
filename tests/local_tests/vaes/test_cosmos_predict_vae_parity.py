# SPDX-License-Identifier: Apache-2.0
"""Component parity test for Cosmos 2.5B Predict VAE."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

REPO_ROOT = Path(__file__).resolve().parents[3]

def _load_official_model(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Load the official component."""
    # Since Cosmos VAE uses AutoencoderKLCosmos from Diffusers, we import it directly.
    from diffusers.models.autoencoders.autoencoder_kl_cosmos import AutoencoderKLCosmos
    
    # Instantiate the official model. The config arguments should match the Cosmos 2.5B Predict VAE.
    model = AutoencoderKLCosmos(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        encoder_block_out_channels=(128, 256, 512, 512),
        decode_block_out_channels=(256, 512, 512, 512),
        spatial_compression_ratio=8,
        temporal_compression_ratio=8,
    )
    return model.to(device=device, dtype=dtype).eval()

def _load_fastvideo_model(device: torch.device, dtype: torch.dtype, official: torch.nn.Module) -> torch.nn.Module:
    """Load the FastVideo component."""
    from fastvideo.models.vaes.cosmos25_official_vae import Cosmos25VAE

    # We instantiate the FastVideo port of the Cosmos 2.5 VAE
    model = Cosmos25VAE()
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model

def _make_inputs(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Create deterministic inputs."""
    torch.manual_seed(0)
    # The Cosmos VAE takes video inputs of shape (batch, channels, frames, height, width)
    return {
        "sample": torch.randn(1, 3, 9, 32, 32, device=device, dtype=dtype),
    }

def _run_official(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        # Encode then decode to test the full pipeline
        latent_dist = model.encode(inputs["sample"]).latent_dist
        z = latent_dist.mode()
        out = model.decode(z).sample
    return out.detach().float().cpu()

def _run_fastvideo(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        z = model.encode(inputs["sample"])
        out = model.decode(z)
    return out.detach().float().cpu()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this parity test.")
def test_component_parity():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    official = _load_official_model(device, dtype)
    fastvideo = _load_fastvideo_model(device, dtype, official)
    
    # The state dict keys differ significantly (diffusers vs native).
    # Since we want to test architectural parity, we can try to copy parameters sequentially.
    with torch.no_grad():
        official_params = list(official.parameters())
        fastvideo_params = list(fastvideo.parameters())
        if len(official_params) == len(fastvideo_params):
            for p1, p2 in zip(official_params, fastvideo_params):
                if p1.shape == p2.shape:
                    p2.copy_(p1)
                else:
                    print(f"Shape mismatch: {p1.shape} vs {p2.shape}")
        else:
            print(f"Param count mismatch: {len(official_params)} vs {len(fastvideo_params)}")


    inputs = _make_inputs(device, dtype)

    official_out = _run_official(official, inputs)
    fastvideo_out = _run_fastvideo(fastvideo, inputs)

    assert official_out.shape == fastvideo_out.shape
    diff = (official_out - fastvideo_out).abs()
    print(f"official abs_mean={official_out.abs().mean().item():.6f} "
          f"fastvideo abs_mean={fastvideo_out.abs().mean().item():.6f} "
          f"diff_max={diff.max().item():.6f} diff_mean={diff.mean().item():.6f}")

    assert_close(fastvideo_out, official_out, atol=5e-2, rtol=5e-2)
