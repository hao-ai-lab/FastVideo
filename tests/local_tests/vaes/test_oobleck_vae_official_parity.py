# SPDX-License-Identifier: Apache-2.0
"""Parity test: FastVideo's `OobleckVAE` port vs the official
`stable_audio_tools.models.autoencoders.OobleckEncoder/OobleckDecoder`
from Stability-AI/stable-audio-tools.

This is the skill-compliant parity test (per `add-model` step 5: "clone
the official reference repo locally"). The companion test in this same
directory (`test_oobleck_vae_parity.py`) compares against
`diffusers.AutoencoderOobleck`, which is the canonical *runtime loader*
for the published HF weights and is already exact (diff=0). This test
adds the *structural* parity check against the official Python source.

What it does:
  1. Build both encoders/decoders with the diffusers-shape config that
     matches what Stable Audio Open 1.0 was trained for (5 downsampling
     stages, latent_dim=64 / params 128, snake activations, no tanh).
  2. Walk both state_dicts in declaration order — they have the same
     183 parameters in the same order, modulo a (C,) vs (1, C, 1)
     storage-shape difference for Snake alpha/beta.
  3. Copy upstream's randomly-initialized weights into FastVideo's
     module via positional pairing + shape-reshape where needed.
  4. Forward identical input through both, assert close.

Skips when:
  * CUDA is unavailable.
  * `stable-audio-tools` isn't importable (clone missing or its
    `alias_free_torch` dep isn't installed). Install via:
        git clone --depth 1 https://github.com/Stability-AI/stable-audio-tools.git
        pip install --no-deps -e ./stable-audio-tools
        pip install alias_free_torch
"""
from __future__ import annotations

import os

import pytest
import torch
from torch.testing import assert_close


def _stable_audio_tools_available() -> bool:
    try:
        from stable_audio_tools.models.autoencoders import OobleckEncoder  # noqa
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Oobleck official parity test requires CUDA.",
)
@pytest.mark.skipif(
    not _stable_audio_tools_available(),
    reason=(
        "Official `stable-audio-tools` not importable. Install with:\n"
        "  git clone --depth 1 https://github.com/Stability-AI/stable-audio-tools.git\n"
        "  pip install --no-deps -e ./stable-audio-tools\n"
        "  pip install alias_free_torch"
    ),
)
def test_oobleck_official_parity():
    from stable_audio_tools.models.autoencoders import OobleckDecoder, OobleckEncoder
    from fastvideo.models.vaes.oobleck import OobleckVAE

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    # Build upstream with the diffusers-shape config that matches
    # stabilityai/stable-audio-open-1.0/vae/config.json:
    #   downsampling_ratios=[2,4,4,8,8] -> upstream strides=[2,4,4,8,8]
    #   channel_multiples=[1,2,4,8,16]   -> upstream c_mults=[1,2,4,8,16]
    #   decoder_input_channels=64        -> upstream encoder latent_dim=128 (mean+scale chunks)
    #                                                    decoder latent_dim=64
    #   audio_channels=2 (stereo)
    #   snake activations on, no tanh on decoder
    up_enc = OobleckEncoder(
        in_channels=2, channels=128, latent_dim=128,
        c_mults=[1, 2, 4, 8, 16], strides=[2, 4, 4, 8, 8],
        use_snake=True, antialias_activation=False,
    ).to(device).eval()
    up_dec = OobleckDecoder(
        out_channels=2, channels=128, latent_dim=64,
        c_mults=[1, 2, 4, 8, 16], strides=[2, 4, 4, 8, 8],
        use_snake=True, antialias_activation=False, final_tanh=False,
    ).to(device).eval()

    # FastVideo OobleckVAE — extract its encoder/decoder.
    fv_vae = OobleckVAE(
        encoder_hidden_size=128,
        downsampling_ratios=[2, 4, 4, 8, 8],
        channel_multiples=[1, 2, 4, 8, 16],
        decoder_channels=128, decoder_input_channels=64,
        audio_channels=2, sampling_rate=44100,
    ).to(device).eval()

    # Transfer upstream weights into FastVideo via positional pairing.
    _transfer_state_positional(up_enc, fv_vae.encoder, "encoder")
    _transfer_state_positional(up_dec, fv_vae.decoder, "decoder")

    # Forward identical input.
    sr = fv_vae.sampling_rate
    n = (sr // fv_vae.hop_length) * fv_vae.hop_length  # full chunks only
    waveform = torch.randn((1, fv_vae.audio_channels, n),
                           dtype=torch.float32, device=device)

    with torch.inference_mode():
        # Encoder parity: upstream returns raw conv output (mean+scale
        # concatenated along channel); FastVideo's encoder.forward returns
        # the same. Diffusers wraps it in a posterior, but here we test
        # the raw conv stack.
        up_enc_out = up_enc(waveform).detach().float().cpu()
        fv_enc_out = fv_vae.encoder(waveform).detach().float().cpu()

    print(
        f"encoder: up shape={tuple(up_enc_out.shape)} abs_mean={up_enc_out.abs().mean().item():.6f}"
    )
    print(
        f"encoder: fv shape={tuple(fv_enc_out.shape)} abs_mean={fv_enc_out.abs().mean().item():.6f}"
    )
    enc_diff = (up_enc_out - fv_enc_out).abs()
    print(f"encoder diff max={enc_diff.max().item():.6e} mean={enc_diff.mean().item():.6e}")

    assert up_enc_out.shape == fv_enc_out.shape
    assert_close(fv_enc_out, up_enc_out, atol=1e-5, rtol=1e-5)

    # Decoder parity on a random latent of the right shape.
    torch.manual_seed(1)
    latent = torch.randn((1, fv_vae.decoder_input_channels, 4),
                         dtype=torch.float32, device=device)
    with torch.inference_mode():
        up_dec_out = up_dec(latent).detach().float().cpu()
        fv_dec_out = fv_vae.decoder(latent).detach().float().cpu()

    print(
        f"decoder: up shape={tuple(up_dec_out.shape)} abs_mean={up_dec_out.abs().mean().item():.6f}"
    )
    print(
        f"decoder: fv shape={tuple(fv_dec_out.shape)} abs_mean={fv_dec_out.abs().mean().item():.6f}"
    )
    dec_diff = (up_dec_out - fv_dec_out).abs()
    print(f"decoder diff max={dec_diff.max().item():.6e} mean={dec_diff.mean().item():.6e}")

    assert up_dec_out.shape == fv_dec_out.shape
    assert_close(fv_dec_out, up_dec_out, atol=1e-5, rtol=1e-5)


def _transfer_state_positional(src: torch.nn.Module, dst: torch.nn.Module, label: str) -> None:
    """Copy `src.state_dict()` into `dst.state_dict()` by walking both in
    declaration order. Reshapes Snake alpha/beta from (C,) to (1, C, 1)
    where needed (the only structural difference between upstream and
    FastVideo's port).

    Asserts that both sides have the same number of parameters and that
    each pair has the same number of elements.
    """
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    src_items = list(src_state.items())
    dst_items = list(dst_state.items())
    assert len(src_items) == len(dst_items), (
        f"{label}: param count mismatch — src={len(src_items)} dst={len(dst_items)}"
    )
    new_state: dict[str, torch.Tensor] = {}
    for (sk, sv), (dk, dv) in zip(src_items, dst_items):
        if sv.shape == dv.shape:
            new_state[dk] = sv.clone()
        elif sv.numel() == dv.numel():
            new_state[dk] = sv.reshape(dv.shape).clone()
        else:
            raise RuntimeError(
                f"{label}: numel mismatch at src={sk}{tuple(sv.shape)} "
                f"vs dst={dk}{tuple(dv.shape)}"
            )
    missing, unexpected = dst.load_state_dict(new_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"{label}: load_state_dict reported missing={missing[:3]} unexpected={unexpected[:3]}"
        )
