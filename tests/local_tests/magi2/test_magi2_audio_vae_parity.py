# SPDX-License-Identifier: Apache-2.0
"""Strict Stable Audio VAE decode parity for MAGI-2 Preview.

Coverage scope: implementation_subcomponent. The official side instantiates
``inference.pipeline.audio_decoder.SAAudioFeatureExtractor``, which builds the
official ``AudioAutoencoder`` from ``inference.model.sa_audio_vae``. The
FastVideo side uses ``fastvideo.models.vaes.magi2_audio_vae.Magi2AudioVAE``,
which wraps ``fastvideo.models.vaes.oobleck.OobleckDecoder``. The test also
checks MAGI-2's latent transpose, sample-major stereo layout, and 441/512
resampling contract.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.models.vaes.magi2_audio_vae import load_magi2_audio_vae
from fastvideo.pipelines.basic.magi2.stages.audio_decoding import (
    decode_magi2_audio,
)
from tests.local_tests.magi2._parity_utils import (
    LOCAL_WEIGHTS_DIR,
    assert_array_exact,
    assert_tensor_exact,
    import_official_module,
    require_path,
)


PARITY_COVERAGE = "both"
STABLE_AUDIO_DIR = LOCAL_WEIGHTS_DIR / "stable-audio-open-1.0"
STABLE_AUDIO_CONFIG_PATH = STABLE_AUDIO_DIR / "model_config.json"
STABLE_AUDIO_CHECKPOINT_PATH = STABLE_AUDIO_DIR / "model.safetensors"


def _deterministic_audio_latent(device: torch.device) -> torch.Tensor:
    """Create a short sample-major latent with MAGI-2's 64 audio channels."""
    latent_values = torch.arange(8 * 64, dtype=torch.float32)
    latent_values = ((latent_values % 127) - 63) / 64
    return latent_values.reshape(8, 64).to(device=device)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MAGI-2 Stable Audio VAE parity requires CUDA.",
)
def test_magi2_stable_audio_decode_and_resample_exact_parity() -> None:
    """Require exact decoded waveform and sample-major resampled audio."""
    require_path(STABLE_AUDIO_CONFIG_PATH, "Stable Audio configuration")
    require_path(STABLE_AUDIO_CHECKPOINT_PATH, "Stable Audio checkpoint")
    official_audio_module = import_official_module(
        "inference.pipeline.audio_decoder"
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    official_audio_vae = official_audio_module.SAAudioFeatureExtractor(
        str(STABLE_AUDIO_DIR)
    )
    fastvideo_audio_vae = load_magi2_audio_vae(STABLE_AUDIO_DIR, device)
    assert official_audio_vae.sample_rate == fastvideo_audio_vae.sampling_rate == 44100
    assert official_audio_vae.downsampling_ratio == fastvideo_audio_vae.hop_length == 2048

    sample_major_latent = _deterministic_audio_latent(device)
    official_channel_major_latent = sample_major_latent.T
    fastvideo_channel_major_latent = sample_major_latent.unsqueeze(0).permute(0, 2, 1).contiguous()
    with torch.inference_mode():
        official_waveform = official_audio_vae.decode(official_channel_major_latent).detach().cpu()
        fastvideo_waveform = fastvideo_audio_vae.decode(
            fastvideo_channel_major_latent
        ).detach().cpu()

    assert_tensor_exact(fastvideo_waveform, official_waveform, "decoded audio waveform")
    official_sample_major = official_waveform.squeeze(0).T.numpy()
    fastvideo_sample_major = fastvideo_waveform.squeeze(0).T.numpy()
    assert_array_exact(fastvideo_sample_major, official_sample_major, "sample-major stereo audio")

    official_resampled = official_audio_module.resample_audio_sinc(
        official_sample_major,
        441 / 512,
    )
    fastvideo_resampled = decode_magi2_audio(
        fastvideo_audio_vae,
        sample_major_latent,
    )
    assert_array_exact(fastvideo_resampled, official_resampled, "resampled stereo audio")
