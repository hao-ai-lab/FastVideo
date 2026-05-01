# SPDX-License-Identifier: Apache-2.0
"""Parity test: FastVideo audio-to-audio variation vs the official
`Stability-AI/stable-audio-tools.generate_diffusion_cond(init_audio=...,
init_noise_level=...)`.

Same harness as `test_stable_audio_pipeline_parity.py` — drives both
sides with the same prompt / seed / steps / CFG / init reference, then
compares decoded waveforms.
"""
from __future__ import annotations

import json

import pytest
import torch

from fastvideo.utils import resolve_hf_token
from ._stable_audio_helpers import (
    can_access_repo,
    setup_hf_env,
)

_HF_REPO_ID = "stabilityai/stable-audio-open-1.0"  # raw upstream weights for the official side
_FV_REPO_ID = "FastVideo/stable-audio-open-1.0-Diffusers"  # converted Diffusers layout for FastVideo's loader
_MODEL_CFG = "model_config.json"
_MODEL_WEIGHTS = "model.safetensors"


def _stable_audio_tools_inference_available() -> bool:
    try:
        from stable_audio_tools.inference.generation import generate_diffusion_cond  # noqa
        from stable_audio_tools.models.factory import create_model_from_config  # noqa
        return True
    except Exception:
        return False


def _load_official(device):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config

    cfg_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_MODEL_CFG, token=resolve_hf_token())
    weights_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_MODEL_WEIGHTS,
                                   token=resolve_hf_token())
    with open(cfg_path) as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    state = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return model.to(device).eval(), model_config


def _make_init_audio(seed: int, sample_rate: int, seconds: float, channels: int = 2):
    """Deterministic synthetic reference: a stereo sine sweep."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = int(seconds * sample_rate)
    t = torch.linspace(0, n / sample_rate, n)
    freqs = torch.linspace(220.0, 880.0, n)
    base = torch.sin(2 * torch.pi * freqs * t) * 0.3
    # Add a touch of noise so the encoder sees nontrivial information.
    base = base + torch.randn(n, generator=g) * 0.01
    return base.unsqueeze(0).repeat(channels, 1).unsqueeze(0).contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Stable Audio A2A parity requires CUDA.")
@pytest.mark.skipif(not can_access_repo(_HF_REPO_ID),
                    reason=f"{_HF_REPO_ID} not accessible — gated.")
@pytest.mark.skipif(not _stable_audio_tools_inference_available(),
                    reason="`stable_audio_tools.inference` not importable.")
def test_stable_audio_a2a_official_parity():
    setup_hf_env()
    device = torch.device("cuda:0")

    prompt = "A gentle ambient pad with soft synth swells."
    seed = 0
    num_inference_steps = 25
    guidance_scale = 7.0
    audio_end_in_s = 1.5
    init_noise_level = 1.0

    off_model, off_cfg = _load_official(device)
    sample_rate = int(off_cfg["sample_rate"])
    full_sample_size = int(off_cfg["sample_size"])

    init_audio = _make_init_audio(seed=seed, sample_rate=sample_rate,
                                  seconds=audio_end_in_s)

    cond = [{"prompt": prompt, "seconds_start": 0.0, "seconds_total": audio_end_in_s}]

    from stable_audio_tools.inference.generation import generate_diffusion_cond
    audio_full = generate_diffusion_cond(
        off_model,
        steps=num_inference_steps,
        cfg_scale=guidance_scale,
        conditioning=cond,
        sample_size=full_sample_size,
        seed=seed,
        device="cuda",
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.3,
        sigma_max=500,
        init_audio=(sample_rate, init_audio.squeeze(0)),
        init_noise_level=init_noise_level,
    )
    end_idx = int(audio_end_in_s * sample_rate)
    off_audio = audio_full.detach().float().cpu()[:, :, :end_idx]
    print(f"off shape={tuple(off_audio.shape)} abs_mean={off_audio.abs().mean():.6f}")

    del off_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    from fastvideo import VideoGenerator
    generator = VideoGenerator.from_pretrained(
        _FV_REPO_ID,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
    )
    try:
        result = generator.generate_video(
            prompt=prompt,
            output_path="outputs_audio/sa_a2a_parity.mp4",
            save_video=False,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            audio_end_in_s=audio_end_in_s,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
        )
    finally:
        generator.shutdown()

    fv_audio = result.get("audio")
    if fv_audio is None:
        ext = result.get("extra", {}) or {}
        fv_audio = ext.get("decoded_audio")
    assert fv_audio is not None
    if not torch.is_tensor(fv_audio):
        import numpy as np
        fv_audio = torch.from_numpy(np.asarray(fv_audio))
    fv_audio = fv_audio.detach().float().cpu()
    if fv_audio.ndim == 2 and fv_audio.shape[0] != 1:
        fv_audio = fv_audio.T.unsqueeze(0)
    print(f"fv  shape={tuple(fv_audio.shape)} abs_mean={fv_audio.abs().mean():.6f}")

    assert fv_audio.shape == off_audio.shape
    diff = (fv_audio - off_audio).abs()
    diff_max = diff.max().item()
    diff_mean = diff.mean().item()
    diff_median = diff.median().item()
    rel = abs(off_audio.abs().mean() - fv_audio.abs().mean()) / max(off_audio.abs().mean().item(),
                                                                    1e-6)
    print(f"diff max={diff_max:.6f} mean={diff_mean:.6f} median={diff_median:.6f} "
          f"abs_mean rel drift={rel:.4%}")

    # A2A is more sensitive than T2A — the encode-side
    # `posterior.sample()` adds per-element randn noise, so even tiny
    # RNG-ordering differences propagate through the SDE trajectory.
    # Empirically (after hoisting the CFG batching + null_embed out of
    # the sampler loop): drift ~0.5%, mean diff ~0.003.
    assert rel < 0.02, f"A2A abs_mean drift {rel:.2%} > 2%"
    assert diff_mean < 0.05, f"A2A mean diff {diff_mean:.4f} > 0.05"
    # Both waveforms should stay in the audio range (no NaN, no blow-up).
    assert torch.isfinite(fv_audio).all(), "FV A2A produced non-finite values"
    assert fv_audio.abs().max().item() < 5.0, "FV A2A magnitude blew up"
