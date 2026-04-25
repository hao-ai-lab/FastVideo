# SPDX-License-Identifier: Apache-2.0
"""Parity test: FastVideo's first-class `OobleckVAE` port vs
`diffusers.AutoencoderOobleck`.

Both classes should produce bit-identical outputs on the same inputs
when loaded from the same published Stable Audio Open 1.0 weights —
FastVideo's port is a 1:1 structural rewrite of Diffusers' class (no
Diffusers Mixin baggage) so diffs are expected to be zero on fp32.

Skips when:
  * CUDA is unavailable (keeps the test aligned with FastVideo's
    GPU-only runtime).
  * `stabilityai/stable-audio-open-1.0` is inaccessible (gated; caller
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
    reason="Oobleck VAE parity test requires CUDA.",
)
@pytest.mark.skipif(
    not _can_access(),
    reason=(f"{_SA_AUDIO_ID} not accessible — gated Stability AI repo; "
            "set HF_TOKEN / HF_API_KEY and accept the terms on "
            f"https://huggingface.co/{_SA_AUDIO_ID}."),
)
def test_oobleck_vae_decode_parity_with_diffusers():
    # Make sure HF_TOKEN is set to whichever alias is actually present.
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(src)
        if v:
            os.environ.setdefault("HF_TOKEN", v)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", v)
            break

    device = torch.device("cuda:0")

    # --- Reference: Diffusers' AutoencoderOobleck ---
    from diffusers import AutoencoderOobleck
    ref_vae = AutoencoderOobleck.from_pretrained(
        _SA_AUDIO_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()

    # --- FastVideo first-class port ---
    from fastvideo.models.vaes.oobleck import OobleckVAE
    fv_vae = OobleckVAE.from_pretrained(
        _SA_AUDIO_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()

    # --- Tiny shared latent ---
    torch.manual_seed(0)
    latent = torch.randn(
        (1, fv_vae.decoder_input_channels, 8),
        dtype=torch.float32, device=device,
    )

    with torch.inference_mode():
        ref_out = ref_vae.decode(latent).sample.detach().float().cpu()
        fv_out = fv_vae.decode(latent).sample.detach().float().cpu()

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
    # Both sides run the same architecture on the same weights in fp32 —
    # should agree to machine epsilon.
    assert_close(fv_out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Oobleck VAE parity test requires CUDA.",
)
@pytest.mark.skipif(
    not _can_access(),
    reason=(f"{_SA_AUDIO_ID} not accessible — gated Stability AI repo; "
            "set HF_TOKEN / HF_API_KEY and accept the terms on "
            f"https://huggingface.co/{_SA_AUDIO_ID}."),
)
def test_oobleck_vae_encode_parity_with_diffusers():
    """Encode-side parity: reuses the deterministic mode of the diagonal
    Gaussian posterior (no sampling).
    """
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(src)
        if v:
            os.environ.setdefault("HF_TOKEN", v)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", v)
            break

    device = torch.device("cuda:0")

    from diffusers import AutoencoderOobleck
    ref_vae = AutoencoderOobleck.from_pretrained(
        _SA_AUDIO_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()

    from fastvideo.models.vaes.oobleck import OobleckVAE
    fv_vae = OobleckVAE.from_pretrained(
        _SA_AUDIO_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()

    torch.manual_seed(1)
    # Stereo, 1 second @ 44100 Hz.
    waveform = torch.randn(
        (1, fv_vae.audio_channels, 44100),
        dtype=torch.float32, device=device,
    )

    with torch.inference_mode():
        ref_dist = ref_vae.encode(waveform).latent_dist
        fv_dist = fv_vae.encode(waveform)
        ref_mode = ref_dist.mode().detach().float().cpu()
        fv_mode = fv_dist.mode().detach().float().cpu()

    diff = (ref_mode - fv_mode).abs()
    print(
        f"ref_mode shape={tuple(ref_mode.shape)} abs_mean={ref_mode.abs().mean().item():.6f}"
    )
    print(
        f"fv_mode  shape={tuple(fv_mode.shape)} abs_mean={fv_mode.abs().mean().item():.6f}"
    )
    print(f"diff max={diff.max().item():.6e} mean={diff.mean().item():.6e}")

    assert ref_mode.shape == fv_mode.shape
    assert_close(fv_mode, ref_mode, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Oobleck VAE round-trip test requires CUDA.",
)
@pytest.mark.skipif(
    not _can_access(),
    reason=(f"{_SA_AUDIO_ID} not accessible — gated Stability AI repo; "
            "set HF_TOKEN / HF_API_KEY and accept the terms on "
            f"https://huggingface.co/{_SA_AUDIO_ID}."),
)
def test_oobleck_vae_round_trip_sanity():
    """Sanity check: encode then decode a real-ish waveform and verify
    the output is bounded and structurally similar (not identity — VAEs
    lose information — but within a sensible reconstruction band).
    """
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(src)
        if v:
            os.environ.setdefault("HF_TOKEN", v)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", v)
            break

    device = torch.device("cuda:0")
    from fastvideo.models.vaes.oobleck import OobleckVAE
    fv_vae = OobleckVAE.from_pretrained(
        _SA_AUDIO_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()

    torch.manual_seed(2)
    # Build ~1 second of mono-mixed-into-stereo sine sweep. Round to a
    # multiple of the VAE's hop_length (= product of downsampling
    # ratios = 2*4*4*8*8 = 2048) so encode → decode preserves length
    # exactly; otherwise the VAE silently drops the trailing partial
    # chunk and the shape assertion below would fail on a non-bug.
    sr = fv_vae.sampling_rate
    n = (sr // fv_vae.hop_length) * fv_vae.hop_length
    t = torch.linspace(0, n / sr, n, device=device)
    freqs = torch.linspace(220.0, 880.0, n, device=device)  # sweep
    mono = torch.sin(2 * torch.pi * freqs * t) * 0.3
    waveform = mono.unsqueeze(0).repeat(2, 1).unsqueeze(0).contiguous()  # [1, 2, n]

    with torch.inference_mode():
        latent = fv_vae.encode(waveform).mode()
        recon = fv_vae.decode(latent).sample

    assert recon.shape == waveform.shape, (
        f"round-trip shape mismatch: in={waveform.shape} out={recon.shape}"
    )
    # Values should stay in a sane range for audio (no NaN / runaway).
    assert torch.isfinite(recon).all(), "round-trip produced non-finite values"
    assert recon.abs().max().item() < 5.0, (
        f"round-trip output magnitude {recon.abs().max().item():.3f} > 5.0 — "
        "VAE may be diverging on synthetic input."
    )
    # The reconstruction should at least correlate with the input on
    # mean-power (diffusion VAEs aren't bit-perfect but they do
    # preserve gross signal energy).
    in_rms = waveform.float().pow(2).mean().sqrt().item()
    out_rms = recon.float().pow(2).mean().sqrt().item()
    rms_ratio = out_rms / max(in_rms, 1e-6)
    print(
        f"round-trip in_rms={in_rms:.4f} out_rms={out_rms:.4f} "
        f"ratio={rms_ratio:.3f}"
    )
    # 0.3x to 3x is generous; outside this band suggests something is
    # broken (silent output or runaway gain).
    assert 0.3 < rms_ratio < 3.0, (
        f"round-trip RMS ratio {rms_ratio:.3f} outside sanity band [0.3, 3.0]"
    )
