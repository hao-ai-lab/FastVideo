# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Open-H3-Regenerate-2K DGX Spark opt-ins: Sol-Attn step schedules and the fast VAE decoder."""

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.models.dits import minimax_h3_sol_attn as sol
from fastvideo.models.vaes.minimax_h3_fast_vae import FastH3VaeDecoder
from fastvideo.models.vaes.minimax_h3_fast_vae_decoder import NonCausalViTDecoder


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("off", [None, None, None]),
        ("sol_engine", [None, 1.0, 1.0]),
        ("sol_engine_fast", [1.75, 1.75, 1.75]),
        ("sol_engine_tau0", [0.0, 1.0, 1.0]),
        ("sol_engine_fast_tau0", [0.0, 1.75, 1.75]),
    ],
)
def test_sol_attn_presets_schedule_each_refine_step(preset, expected):
    with envs.FASTVIDEO_H3_SOL_ATTN.override(preset):
        got = [sol.tau(step) if sol.sparse_step(step) else None for step in range(3)]
        assert got == expected
        # Steps past the schedule reuse its last entry.
        assert sol.sparse_step(7) == (expected[-1] is not None)


def test_sol_attn_rejects_unknown_preset():
    with envs.FASTVIDEO_H3_SOL_ATTN.override("sol_engine_turbo"), pytest.raises(envs.EnvVarError):
        sol.mode()


def test_sol_attn_sink_tokens_are_clamped():
    sol.set_sink_tokens(-3)
    assert sol.get_sink_tokens() == 0
    sol.set_sink_tokens(452)
    assert sol.get_sink_tokens() == 452


def _tiny_fast_decoder() -> FastH3VaeDecoder:
    model = NonCausalViTDecoder([0.0] * 24, [1.0] * 24, list(range(2)), time_norm=5, heads=2, dim_head=8)
    decoder = FastH3VaeDecoder.__new__(FastH3VaeDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.model = model.eval()
    decoder._forward = decoder.model
    decoder.latent_channels = 24
    decoder.spatial_compression_ratio = 16
    decoder.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1, 1), persistent=False)
    decoder.register_buffer("pixel_std", torch.ones(1, 3, 1, 1, 1), persistent=False)
    return decoder


@pytest.mark.parametrize("latent_frames", [5, 7, 10])
def test_fast_vae_pixel_shape_matches_the_decoder(latent_frames):
    decoder = _tiny_fast_decoder()
    latents = torch.randn(1, 24, latent_frames, 2, 3)
    z = decoder.denormalize_latents(latents)
    assert z.shape == (1, latent_frames, 24, 2, 3)
    expected = decoder.decoded_pixel_shape(z.shape)
    with torch.no_grad():
        pixels = decoder.model(z.float())
    assert tuple(pixels.shape) == expected
    # 5 latent frames are one native block of 17 video frames.
    assert expected[2] == latent_frames * 4 - 3 * ((latent_frames + 4) // 5)


def test_fast_vae_rejects_wrong_channel_count():
    decoder = _tiny_fast_decoder()
    with pytest.raises(ValueError, match="24 channels"):
        decoder.decoded_pixel_shape((1, 5, 16, 2, 3))
