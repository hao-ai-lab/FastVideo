# SPDX-License-Identifier: Apache-2.0
"""Decode parity for the reused VAE: FastVideo's ``AutoencoderKL`` (production
``VAELoader``, decoder-only as ``OvisImageT2IConfig`` configures it) vs the
official Diffusers ``AutoencoderKL``, decoding the same raw latent."""

import os

import pytest
import torch

import fastvideo  # noqa: F401  # ensure full package init before deep submodule imports
from fastvideo.configs.pipelines.ovis_image import OvisImageT2IConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import VAELoader

logger = init_logger(__name__)

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29511")

LOCAL_WEIGHTS = os.getenv("OVIS_WEIGHTS", "official_weights/ovis_image")
VAE_PATH = os.path.join(LOCAL_WEIGHTS, "vae")


def _decode(vae, z):
    """Decode a latent through either a FastVideo or Diffusers VAE module."""
    out = vae.decode(z)
    if hasattr(out, "sample"):
        return out.sample
    return out[0] if isinstance(out, (tuple, list)) else out


@pytest.mark.skipif(
    not os.path.exists(VAE_PATH),
    reason=(f"Ovis-Image VAE not found at {VAE_PATH}. "
            f"Set OVIS_WEIGHTS or download from AIDC-AI/Ovis-Image-7B."))
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Ovis-Image VAE parity requires CUDA.")
@pytest.mark.usefixtures("distributed_setup")
def test_ovis_vae_decode_parity():
    try:
        from diffusers import AutoencoderKL as RefVAE
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Diffusers AutoencoderKL unavailable: {exc!r}")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    precision = torch.float32  # OvisImageT2IConfig uses fp32 for the VAE

    # ---- FastVideo VAE (production loader, decoder-only via OvisImageT2IConfig) ----
    pipeline_config = OvisImageT2IConfig()
    args = FastVideoArgs(
        model_path=VAE_PATH,
        pipeline_config=pipeline_config,
        pin_cpu_memory=False,
    )
    args.device = device
    fv_vae = VAELoader().load(VAE_PATH, args)
    fv_vae = fv_vae.to(device=device, dtype=precision).eval()

    # ---- Reference VAE (official Diffusers) ----
    ref_vae = RefVAE.from_pretrained(VAE_PATH, torch_dtype=precision).to(
        device).eval()

    z_channels = ref_vae.config.latent_channels
    z = torch.randn(1, z_channels, 32, 32, device=device, dtype=precision)

    with torch.no_grad():
        ref_img = _decode(ref_vae, z)
        fv_img = _decode(fv_vae, z)

    assert ref_img.shape == fv_img.shape, \
        f"Shape mismatch: ref={ref_img.shape}, fv={fv_img.shape}"
    assert torch.isfinite(fv_img).all(), "FastVideo VAE decode has NaN/Inf"
    assert torch.isfinite(ref_img).all(), "Reference VAE decode has NaN/Inf"

    max_diff = (ref_img - fv_img).abs().max().item()
    logger.info(f"VAE decode parity max_diff={max_diff:.3e}")
    # fp32 decode after the same weights — should be near-identical.
    torch.testing.assert_close(ref_img, fv_img, atol=5e-2, rtol=5e-2)
