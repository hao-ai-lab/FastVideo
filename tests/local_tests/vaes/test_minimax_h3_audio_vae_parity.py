# Copyright 2025 MiniMax authors and HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

"""CPU synthetic parity for the native MiniMax H3 audio VAE."""

from __future__ import annotations

import sys

import torch
from torch.testing import assert_close

from tests.local_tests.minimax_h3._reference import REFERENCE_SRC, assert_pinned_reference

assert_pinned_reference(
    "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py",
    "1f57b2072c2450ba475e19fc0c1558c394774d9b3ba2291a62f873267d247745",
)
sys.path.insert(0, str(REFERENCE_SRC))

from diffusers import AutoencoderKLMiniMaxH3Audio as OfficialMiniMaxH3AudioVAE  # noqa: E402

from fastvideo.configs.models.vaes.minimax_h3_audio import (  # noqa: E402
    MiniMaxH3AudioVAEArchConfig,
    MiniMaxH3AudioVAEConfig,
)
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE  # noqa: E402


TINY_ARCH = {
    "encoder_dim": 4,
    "encoder_rates": (2, 2),
    "latent_dim": 32,
    "latent_channels": 8,
    "num_attention_heads": 2,
    "decoder_dim": 16,
    "decoder_rates": (2, 2),
    "decoder_kernel_sizes": (4, 4),
    "resblock_kernel_sizes": (3, 7),
    "resblock_dilation_sizes": ((1, 3), (1, 3)),
    "sampling_rate": 32000,
    "latents_mean": [0.0] * 8,
    "latents_std": [1.0] * 8,
}


def _build_models() -> tuple[OfficialMiniMaxH3AudioVAE, MiniMaxH3AudioVAE]:
    torch.manual_seed(1234)
    official = OfficialMiniMaxH3AudioVAE(**TINY_ARCH).float().eval()

    arch = MiniMaxH3AudioVAEArchConfig(**TINY_ARCH)
    fastvideo = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(arch_config=arch)).eval()

    official_state = official.state_dict()
    fastvideo_state = fastvideo.state_dict()
    assert official_state.keys() == fastvideo_state.keys()
    assert {key: value.shape for key, value in official_state.items()} == {
        key: value.shape for key, value in fastvideo_state.items()
    }
    for key in official_state:
        if key.endswith((".filter", "zero_k_bias")):
            assert_close(fastvideo_state[key], official_state[key], atol=0.0, rtol=0.0)
    fastvideo.load_state_dict(official_state, strict=True)
    return official, fastvideo


def _capture_activations(model: torch.nn.Module, names: tuple[str, ...]) -> tuple[dict[str, list[torch.Tensor]], list]:
    activations: dict[str, list[torch.Tensor]] = {name: [] for name in names}

    def capture(name: str):
        def hook(_module, _inputs, output) -> None:
            assert isinstance(output, torch.Tensor)
            activations[name].append(output.detach().clone())

        return hook

    modules = dict(model.named_modules())
    handles = [modules[name].register_forward_hook(capture(name)) for name in names]
    return activations, handles


def test_minimax_h3_audio_vae_encode_and_decode_match_reference() -> None:
    """Compare posterior tensors, direct decode, and the mode round trip."""

    official, fastvideo = _build_models()
    generator = torch.Generator(device="cpu").manual_seed(2026)
    waveform = torch.randn((2, 1, 33), generator=generator, dtype=torch.float32)
    latents = torch.randn((2, 8, 8), generator=generator, dtype=torch.float32)

    activation_names = ("encoder.block.1", "pre_block", "mean_proj", "decoder.ups.0", "decoder.conv_post")
    official_activations, official_handles = _capture_activations(official, activation_names)
    fastvideo_activations, fastvideo_handles = _capture_activations(fastvideo, activation_names)
    with torch.inference_mode():
        official_posterior = official.encode(waveform, return_dict=False)[0]
        fastvideo_posterior = fastvideo.encode(waveform, return_dict=False)[0]
        official_decode = official.decode(latents).sample
        fastvideo_decode = fastvideo.decode(latents).sample
        official_round_trip = official(waveform, sample_posterior=False, return_dict=False)[0]
        fastvideo_round_trip = fastvideo(waveform, sample_posterior=False, return_dict=False)[0]
        official_generator = torch.Generator(device="cpu").manual_seed(42)
        fastvideo_generator = torch.Generator(device="cpu").manual_seed(42)
        official_sample = official_posterior.sample(generator=official_generator)
        fastvideo_sample = fastvideo_posterior.sample(generator=fastvideo_generator)
    for handle in official_handles + fastvideo_handles:
        handle.remove()

    assert_close(fastvideo_posterior.mode(), official_posterior.mode(), atol=0.0, rtol=0.0)
    assert_close(fastvideo_posterior.logs, official_posterior.logs, atol=0.0, rtol=0.0)
    assert_close(fastvideo_sample, official_sample, atol=0.0, rtol=0.0)
    assert_close(fastvideo_decode, official_decode, atol=0.0, rtol=0.0)
    assert_close(fastvideo_round_trip, official_round_trip, atol=0.0, rtol=0.0)
    assert_close(
        torch.randn(8, generator=fastvideo_generator),
        torch.randn(8, generator=official_generator),
        atol=0,
        rtol=0,
    )

    assert fastvideo.hop_length == 4
    assert fastvideo.sampling_rate == 32000
    assert fastvideo.latent_channels == 8
    assert fastvideo.audio_channels == 1
    assert fastvideo_posterior.mode().shape == (2, 8, 9)
    assert fastvideo_decode.shape == (2, 1, 32)
    assert all(parameter.dtype == torch.float32 for parameter in fastvideo.parameters())
    assert all(buffer.dtype == torch.float32 for buffer in fastvideo.buffers() if buffer.is_floating_point())
    for name in activation_names:
        assert len(fastvideo_activations[name]) == len(official_activations[name])
        for result, expected in zip(fastvideo_activations[name], official_activations[name], strict=True):
            assert_close(result, expected, atol=1e-6, rtol=1e-6)
