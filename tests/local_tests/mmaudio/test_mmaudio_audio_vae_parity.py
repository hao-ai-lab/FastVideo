# SPDX-License-Identifier: Apache-2.0
"""MMAudio 44.1 kHz audio VAE parity against the official reference.

Coverage scope: implementation_subcomponent. Production-loader coverage is
added after the converted component directory exists.
"""

from __future__ import annotations

import gc
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


def _official_model():
    official_vae = pytest.importorskip("mmaudio.ext.autoencoder.vae")
    return official_vae.VAE_44k()


def _fastvideo_model(*, need_encoder: bool):
    from fastvideo.models.audio.mmaudio_vae import MMAudioVAE

    return MMAudioVAE(mode="44k", need_encoder=need_encoder)


def test_mmaudio_44k_audio_vae_state_structure() -> None:
    official = _official_model()
    expected = {name: tensor.shape for name, tensor in official.state_dict().items()}
    del official
    gc.collect()
    fastvideo = _fastvideo_model(need_encoder=True)
    assert expected == {
        name: tensor.shape for name, tensor in fastvideo.state_dict().items()
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mmaudio_44k_audio_vae_decoder_implementation_parity(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("MMAudio audio VAE implementation parity requires CUDA")

    official = _official_model()
    del official.encoder
    gc.collect()
    fastvideo = _fastvideo_model(need_encoder=False)
    fastvideo.load_state_dict(official.state_dict(), strict=True)
    device = torch.device("cuda:0")
    official.remove_weight_norm().to(device=device, dtype=dtype).eval()
    fastvideo.remove_weight_norm().to(device=device, dtype=dtype).eval()
    latent = torch.randn(
        (1, 40, 4),
        generator=torch.Generator(device=device).manual_seed(1234),
        device=device,
        dtype=dtype,
    )

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
    official = _official_model()
    fastvideo = _fastvideo_model(need_encoder=True)
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
