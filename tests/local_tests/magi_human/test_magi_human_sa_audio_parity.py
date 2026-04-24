# SPDX-License-Identifier: Apache-2.0
"""Parity test: FastVideo SAAudioVAEModel wrapper vs direct
`diffusers.AutoencoderOobleck.from_pretrained(...)` on the Stable Audio
Open 1.0 VAE.

Same setup as the T5-Gemma wrapper parity test — both paths are thin
wrappers around the same HF Diffusers class on the same gated repo, so
parity should be exact. Guards against regressions in the lazy-load
device placement, dtype casts, or interface deviations in the wrapper.

Skips when:
  * CUDA is unavailable (VAE is 156M params, small enough for CPU but
    we keep the test GPU-only to match the pipeline's runtime).
  * `stabilityai/stable-audio-open-1.0` is inaccessible (gated; user
    must have accepted terms on the HF repo page).
"""
from __future__ import annotations

import os

import pytest
import torch
from torch.testing import assert_close


_SA_AUDIO_ID = "stabilityai/stable-audio-open-1.0"


def _hf_token():
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def _can_access() -> bool:
    token = _hf_token()
    if token is None:
        return False
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=_SA_AUDIO_ID, filename="vae/config.json", token=token,
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiHuman Stable-Audio VAE parity requires CUDA.",
)
@pytest.mark.skipif(
    not _can_access(),
    reason=(f"{_SA_AUDIO_ID} not accessible — gated Stability AI repo; "
            "set HF_TOKEN / HF_API_KEY and accept the terms on "
            f"https://huggingface.co/{_SA_AUDIO_ID}."),
)
def test_magi_human_sa_audio_vae_decode_parity():
    # Make sure HF_TOKEN is the alias the Diffusers loader actually reads.
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(src)
        if v:
            os.environ.setdefault("HF_TOKEN", v)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", v)
            break

    device = torch.device("cuda:0")

    # --- Reference: direct HF Diffusers call, same as upstream's
    #     SAAudioFeatureExtractor but via the Diffusers Oobleck port. ---
    from diffusers import AutoencoderOobleck
    ref_vae = AutoencoderOobleck.from_pretrained(
        _SA_AUDIO_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()

    # --- FastVideo wrapper path ---
    from fastvideo.configs.models.encoders.sa_audio import SAAudioVAEConfig
    from fastvideo.models.encoders.sa_audio import SAAudioVAEModel
    fv_config = SAAudioVAEConfig()
    fv_config.arch_config.sa_audio_model_path = _SA_AUDIO_ID
    fv_vae = SAAudioVAEModel(fv_config)

    # --- Tiny shared latent ---
    torch.manual_seed(0)
    # decoder_input_channels=64, latent length ~8 frames for a quick test.
    latent = torch.randn(
        (1, fv_config.arch_config.decoder_input_channels, 8),
        dtype=torch.float32, device=device,
    )

    with torch.inference_mode():
        ref_out = ref_vae.decode(latent).sample.detach().float().cpu()
        fv_out = fv_vae.decode(latent).detach().float().cpu()

    print(
        f"ref shape={tuple(ref_out.shape)} "
        f"abs_mean={ref_out.abs().mean().item():.6f} "
        f"range=[{ref_out.min().item():.4f}, {ref_out.max().item():.4f}]"
    )
    print(
        f"fv  shape={tuple(fv_out.shape)} "
        f"abs_mean={fv_out.abs().mean().item():.6f} "
        f"range=[{fv_out.min().item():.4f}, {fv_out.max().item():.4f}]"
    )
    diff = (ref_out - fv_out).abs()
    print(f"diff max={diff.max().item():.6e} mean={diff.mean().item():.6e}")

    assert ref_out.shape == fv_out.shape
    # Both sides call the same HF class on the same weights in fp32 —
    # should agree to machine epsilon.
    assert_close(fv_out, ref_out, atol=1e-5, rtol=1e-5)
