# SPDX-License-Identifier: Apache-2.0
"""MMAudio 44.1 kHz audio VAE parity against the official reference.

Coverage scope: implementation_subcomponent. Production-loader coverage is
added after the converted component directory exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_WEIGHTS = Path(
    os.environ.get(
        "MMAUDIO_AUDIO_VAE_WEIGHTS",
        REPO_ROOT.parent / "MMAudio/ext_weights/v1-44.pth",
    )
)


def _models():
    from mmaudio.ext.autoencoder.vae import VAE_44k

    from fastvideo.models.audio.mmaudio_vae import MMAudioVAE

    return VAE_44k(), MMAudioVAE(mode="44k", need_encoder=True)


def test_mmaudio_44k_audio_vae_state_structure() -> None:
    official, fastvideo = _models()
    assert {name: tensor.shape for name, tensor in official.state_dict().items()} == {
        name: tensor.shape for name, tensor in fastvideo.state_dict().items()
    }


def test_mmaudio_44k_audio_vae_decoder_implementation_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MMAudio audio VAE implementation parity requires CUDA")

    official, fastvideo = _models()
    fastvideo.load_state_dict(official.state_dict(), strict=True)
    # This test exercises decoder math using the same randomly initialized
    # parameters. Drop both unused encoders before moving the large VAE to GPU.
    del official.encoder
    del fastvideo.encoder
    device = torch.device("cuda:0")
    official.remove_weight_norm().to(device).eval()
    fastvideo.remove_weight_norm().to(device).eval()
    latent = torch.randn((1, 40, 4), generator=torch.Generator(device=device).manual_seed(1234), device=device)

    with torch.inference_mode():
        expected = official.decode(latent)
        actual = fastvideo.decode(latent)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_mmaudio_44k_audio_vae_numerical_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MMAudio audio VAE parity requires CUDA")
    if not OFFICIAL_WEIGHTS.is_file():
        pytest.skip(
            "Official MMAudio 44.1 kHz VAE weights are absent. Set MMAUDIO_AUDIO_VAE_WEIGHTS or download v1-44.pth."
        )

    device = torch.device("cuda:0")
    official, fastvideo = _models()
    state = torch.load(OFFICIAL_WEIGHTS, map_location="cpu", weights_only=True)
    official.load_state_dict(state, strict=True)
    fastvideo.load_state_dict(state, strict=True)
    official.remove_weight_norm().to(device).eval()
    fastvideo.remove_weight_norm().to(device).eval()

    generator = torch.Generator(device=device).manual_seed(1234)
    mel = torch.randn((1, 128, 128), generator=generator, device=device)
    latent = torch.randn((1, 40, 64), generator=generator, device=device)

    with torch.inference_mode():
        expected_posterior = official.encode(mel)
        actual_posterior = fastvideo.encode(mel)
        expected_mel = official.decode(latent)
        actual_mel = fastvideo.decode(latent)

    torch.testing.assert_close(actual_posterior.mean, expected_posterior.mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_posterior.logvar, expected_posterior.logvar, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_mel, expected_mel, atol=1e-5, rtol=1e-5)
