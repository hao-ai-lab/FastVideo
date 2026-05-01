# SPDX-License-Identifier: Apache-2.0
"""Parity test: FastVideo MagiHuman Stable-Audio wrapper vs the official
daVinci-MagiHuman Stable-Audio usage path.

The sibling `test_magi_human_sa_audio_parity.py` compares FastVideo's
`SAAudioVAEModel` against Diffusers `AutoencoderOobleck`, which validates the
low-level VAE weights/API. This test instead follows the official
daVinci-MagiHuman integration layer:

* `inference.model.sa_audio.SAAudioFeatureExtractor` is constructed from the
  full Stable Audio checkpoint (`model_config.json` + `model.safetensors`).
* The official loader rebuilds its local `AudioAutoencoder` from
  `model.pretransform.config` and filters `pretransform.model.*` weights.
* The official decode entry point is `SAAudioFeatureExtractor.decode(latents)`,
  which calls `vae_model.decode(latents)` directly. There is no latent
  mean/std normalization or reference-audio injection inside this decode layer.
* Pipeline post-processing is outside the SA module: `MagiEvaluator` transposes
  `[B, L, C] -> [C, L]` before decode, then transposes waveform samples and
  applies `resample_audio_sinc(..., 441 / 512)`.

This catches drift between FastVideo's full SA wrapper path and the official
repo's custom Stable-Audio wrapper/module, not just the bare Diffusers VAE.

Skips when:
  * CUDA is unavailable.
  * `daVinci-MagiHuman/` is not checked out under the repo root.
  * `stabilityai/stable-audio-open-1.0` is inaccessible (gated; user must have
    accepted terms and set HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HF_API_KEY).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close


_SA_AUDIO_ID = "stabilityai/stable-audio-open-1.0"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _upstream_root() -> Path:
    return _repo_root() / "daVinci-MagiHuman"


def _hf_token():
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _can_access() -> bool:
    token = _hf_token()
    if token is None:
        return False
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=_SA_AUDIO_ID,
            filename="model_config.json",
            token=token,
        )
        return True
    except Exception:
        return False


def _stable_audio_snapshot() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=_SA_AUDIO_ID,
        token=_hf_token(),
        allow_patterns=["model_config.json", "model.safetensors"],
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiHuman official Stable-Audio parity requires CUDA.",
)
@pytest.mark.skipif(
    not _upstream_root().exists(),
    reason="daVinci-MagiHuman checkout is required under the repo root.",
)
@pytest.mark.skipif(
    not _can_access(),
    reason=(
        f"{_SA_AUDIO_ID} not accessible — gated Stability AI repo; set "
        "HF_TOKEN / HF_API_KEY and accept the terms on "
        f"https://huggingface.co/{_SA_AUDIO_ID}."
    ),
)
def test_magi_human_sa_audio_official_decode_parity():
    # Make sure both HF helpers and FastVideo's loader see the same token alias.
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        value = os.environ.get(src)
        if value:
            os.environ.setdefault("HF_TOKEN", value)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", value)
            break

    device = torch.device("cuda:0")

    # --- Official daVinci-MagiHuman path: custom SAAudioFeatureExtractor
    #     rebuilds AudioAutoencoder and filters `pretransform.model.*` from
    #     the full Stable Audio checkpoint.
    from tests.local_tests.helpers.magi_human_upstream import install_stubs

    install_stubs()
    from inference.model.sa_audio import SAAudioFeatureExtractor

    upstream_vae = SAAudioFeatureExtractor(
        device=device,
        model_path=_stable_audio_snapshot(),
    )

    # --- FastVideo MagiHuman wrapper path: lazy loader around the native
    #     OobleckVAE port, exactly what the MagiHuman pipeline constructs.
    from fastvideo.configs.models.vaes import OobleckVAEConfig
    from fastvideo.models.vaes.sa_audio import SAAudioVAEModel

    fv_config = OobleckVAEConfig()
    fv_config.pretrained_path = _SA_AUDIO_ID
    fv_config.pretrained_dtype = "float32"
    fv_vae = SAAudioVAEModel(fv_config)

    torch.manual_seed(0)
    latent = torch.randn(
        (1, fv_config.arch_config.decoder_input_channels, 8),
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        upstream_out = upstream_vae.decode(latent).detach().float().cpu()
        fv_out = fv_vae.decode(latent).detach().float().cpu()

    print(
        f"upstream shape={tuple(upstream_out.shape)} "
        f"abs_mean={upstream_out.abs().mean().item():.6f} "
        f"range=[{upstream_out.min().item():.4f}, "
        f"{upstream_out.max().item():.4f}]"
    )
    print(
        f"fv       shape={tuple(fv_out.shape)} "
        f"abs_mean={fv_out.abs().mean().item():.6f} "
        f"range=[{fv_out.min().item():.4f}, {fv_out.max().item():.4f}]"
    )
    diff = (upstream_out - fv_out).abs()
    print(f"diff max={diff.max().item():.6e} mean={diff.mean().item():.6e}")

    assert upstream_out.shape == fv_out.shape
    assert_close(fv_out, upstream_out, atol=1e-5, rtol=1e-5)
