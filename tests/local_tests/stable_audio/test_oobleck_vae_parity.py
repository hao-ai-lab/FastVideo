# SPDX-License-Identifier: Apache-2.0
"""Parity test: FastVideo's first-class `OobleckVAE` port vs the
**official** `Stability-AI/stable-audio-tools` `AudioAutoencoder` loaded
from the published Stable Audio Open 1.0 checkpoint.

(Was previously a diffusers comparison; switched to compare against the
upstream reference per the add-model skill, which calls for parity
against the official repo. The companion test
`test_oobleck_vae_official_parity.py` covers structural parity with
randomly-initialized weights; this file exercises the *real* trained
weights end-to-end through the official factory + checkpoint loader.)

Skips when:
  * CUDA is unavailable.
  * `stabilityai/stable-audio-open-1.0` is inaccessible (gated repo).
  * `stable_audio_tools` is not importable (clone missing or its
    transitive deps not installed — `k_diffusion`, `einops_exts`,
    `alias_free_torch`).

Install official deps once via:
    git clone --depth 1 https://github.com/Stability-AI/stable-audio-tools.git
    uv pip install --no-deps -e ./stable-audio-tools
    uv pip install k_diffusion einops_exts alias_free_torch
"""
from __future__ import annotations

import json

import pytest
import torch
from torch.testing import assert_close

from fastvideo.utils import resolve_hf_token
from ._stable_audio_helpers import (
    can_access_repo,
    setup_hf_env,
)

_SA_AUDIO_ID = "stabilityai/stable-audio-open-1.0"
_VAE_CKPT_FILE = "vae_model.ckpt"
_VAE_CFG_FILE = "vae_model_config.json"


def _stable_audio_tools_available() -> bool:
    try:
        from stable_audio_tools.models.factory import (  # noqa: F401
            create_model_from_config,
        )
        return True
    except Exception:
        return False


def _load_official_vae(device: torch.device):
    """Build the official `AudioAutoencoder` from `vae_model_config.json`
    and load its weights from `vae_model.ckpt`.
    """
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(repo_id=_SA_AUDIO_ID, filename=_VAE_CFG_FILE,
                               token=resolve_hf_token())
    ckpt_path = hf_hub_download(repo_id=_SA_AUDIO_ID, filename=_VAE_CKPT_FILE,
                                token=resolve_hf_token())
    with open(cfg_path) as f:
        vcfg = json.load(f)

    from stable_audio_tools.models.factory import create_model_from_config
    vae = create_model_from_config(vcfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    pref = "autoencoder."
    state = {k[len(pref):] if k.startswith(pref) else k: v for k, v in state.items()}
    missing, unexpected = vae.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (
        f"official VAE state_dict mismatch — missing={missing[:3]} unexpected={unexpected[:3]}"
    )
    return vae.to(device).eval()


_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Oobleck VAE parity test requires CUDA.",
)
_skip_no_access = pytest.mark.skipif(
    not can_access_repo(_SA_AUDIO_ID, filename=_VAE_CFG_FILE),
    reason=(f"{_SA_AUDIO_ID} not accessible — gated repo; set HF_TOKEN / "
            f"HF_API_KEY and accept the terms on https://huggingface.co/{_SA_AUDIO_ID}."),
)
_skip_no_official = pytest.mark.skipif(
    not _stable_audio_tools_available(),
    reason=("`stable_audio_tools` not importable. Clone "
            "https://github.com/Stability-AI/stable-audio-tools and `uv pip install` "
            "its deps (k_diffusion, einops_exts, alias_free_torch)."),
)


@_skip_no_cuda
@_skip_no_access
@_skip_no_official
def test_oobleck_vae_decode_official_parity():
    """Decode-side parity with the official AudioAutoencoder."""
    setup_hf_env()
    device = torch.device("cuda:0")

    off_vae = _load_official_vae(device)

    from fastvideo.models.vaes.oobleck import OobleckVAE
    fv_vae = OobleckVAE.from_pretrained(
        _SA_AUDIO_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()

    torch.manual_seed(0)
    latent = torch.randn(
        (1, fv_vae.decoder_input_channels, 8),
        dtype=torch.float32, device=device,
    )

    with torch.inference_mode():
        off_out = off_vae.decode(latent).detach().float().cpu()
        fv_out = fv_vae.decode(latent).sample.detach().float().cpu()

    print(
        f"off shape={tuple(off_out.shape)} "
        f"abs_mean={off_out.abs().mean().item():.6f} "
        f"range=[{off_out.min().item():.4f}, {off_out.max().item():.4f}]"
    )
    print(
        f"fv  shape={tuple(fv_out.shape)} "
        f"abs_mean={fv_out.abs().mean().item():.6f} "
        f"range=[{fv_out.min().item():.4f}, {fv_out.max().item():.4f}]"
    )
    diff = (off_out - fv_out).abs()
    print(f"diff max={diff.max().item():.6e} mean={diff.mean().item():.6e}")

    assert off_out.shape == fv_out.shape
    # FV is a structural rewrite of upstream's OobleckEncoder/Decoder,
    # loading the same weights via `OobleckVAE.from_pretrained`. Output
    # should be bit-identical at fp32.
    assert_close(fv_out, off_out, atol=1e-5, rtol=1e-5)


@_skip_no_cuda
@_skip_no_access
@_skip_no_official
def test_oobleck_vae_encode_official_parity():
    """Encode-side parity vs the official autoencoder.

    Compares the **mean** of the diagonal Gaussian posterior, not a
    sample. Official `VAEBottleneck.encode` calls `vae_sample(mean,
    scale)` which adds `randn() * softplus(scale)+1e-4` — non-deterministic.
    To compare apples-to-apples we ask the official side to skip the
    bottleneck (`skip_bottleneck=True`), chunk to (mean, scale)
    ourselves, and compare against FV's `.encode().mode()`.
    """
    setup_hf_env()
    device = torch.device("cuda:0")

    off_vae = _load_official_vae(device)

    from fastvideo.models.vaes.oobleck import OobleckVAE
    fv_vae = OobleckVAE.from_pretrained(
        _SA_AUDIO_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()

    torch.manual_seed(1)
    waveform = torch.randn(
        (1, fv_vae.audio_channels, 44100),
        dtype=torch.float32, device=device,
    )
    # Round to a multiple of hop_length so encode aligns.
    n = (waveform.shape[-1] // fv_vae.hop_length) * fv_vae.hop_length
    waveform = waveform[..., :n]

    with torch.inference_mode():
        # Bypass the stochastic VAEBottleneck — we want the mean.
        off_pre = off_vae.encode(waveform, skip_bottleneck=True)
        off_mean, _off_scale = off_pre.chunk(2, dim=1)
        fv_mean = fv_vae.encode(waveform).mode()

    off_mean_cpu = off_mean.detach().float().cpu()
    fv_mean_cpu = fv_mean.detach().float().cpu()

    diff = (off_mean_cpu - fv_mean_cpu).abs()
    print(
        f"off_mean shape={tuple(off_mean_cpu.shape)} abs_mean={off_mean_cpu.abs().mean().item():.6f}"
    )
    print(
        f"fv_mean  shape={tuple(fv_mean_cpu.shape)} abs_mean={fv_mean_cpu.abs().mean().item():.6f}"
    )
    print(f"diff max={diff.max().item():.6e} mean={diff.mean().item():.6e}")

    assert off_mean_cpu.shape == fv_mean_cpu.shape
    assert_close(fv_mean_cpu, off_mean_cpu, atol=1e-5, rtol=1e-5)


@_skip_no_cuda
@_skip_no_access
@_skip_no_official
def test_oobleck_vae_round_trip_sanity():
    """Sanity check: encode → decode preserves shape and stays in a
    sensible amplitude range. Not a parity test — just guards against
    silent diverging VAE output. Uses FV alone (no comparison).
    """
    setup_hf_env()
    device = torch.device("cuda:0")

    from fastvideo.models.vaes.oobleck import OobleckVAE
    fv_vae = OobleckVAE.from_pretrained(
        _SA_AUDIO_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()

    torch.manual_seed(2)
    sr = fv_vae.sampling_rate
    n = (sr // fv_vae.hop_length) * fv_vae.hop_length
    t = torch.linspace(0, n / sr, n, device=device)
    freqs = torch.linspace(220.0, 880.0, n, device=device)
    mono = torch.sin(2 * torch.pi * freqs * t) * 0.3
    waveform = mono.unsqueeze(0).repeat(2, 1).unsqueeze(0).contiguous()

    with torch.inference_mode():
        latent = fv_vae.encode(waveform).mode()
        recon = fv_vae.decode(latent).sample

    assert recon.shape == waveform.shape, (
        f"round-trip shape mismatch: in={waveform.shape} out={recon.shape}"
    )
    assert torch.isfinite(recon).all(), "round-trip produced non-finite values"
    assert recon.abs().max().item() < 5.0, (
        f"round-trip output magnitude {recon.abs().max().item():.3f} > 5.0"
    )

    in_rms = waveform.float().pow(2).mean().sqrt().item()
    out_rms = recon.float().pow(2).mean().sqrt().item()
    rms_ratio = out_rms / max(in_rms, 1e-6)
    print(
        f"round-trip in_rms={in_rms:.4f} out_rms={out_rms:.4f} "
        f"ratio={rms_ratio:.3f}"
    )
    assert 0.3 < rms_ratio < 3.0, (
        f"round-trip RMS ratio {rms_ratio:.3f} outside sanity band [0.3, 3.0]"
    )
