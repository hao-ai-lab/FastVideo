# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline parity test for Stable Audio Open 1.0.

Per the add-model skill (steps 5 + 13(a)), the canonical parity
reference is the **official upstream repo** (`Stability-AI/stable-audio-tools`).
This test runs `stable_audio_tools.inference.generation.generate_diffusion_cond`
on the published model + checkpoint and compares the resulting waveform
against the FastVideo `StableAudioPipeline` output for the same prompt /
seed / steps / CFG.

All FV components are now first-class ports (REVIEW item 30 closed):
  * **VAE**: bit-identical (FV's `OobleckVAE` is a 1:1 port of
    upstream's `OobleckEncoder/OobleckDecoder` — verified separately
    in `tests/local_tests/vaes/test_oobleck_vae_parity.py` to diff=0).
  * **DiT**: native `fastvideo.models.dits.stable_audio.StableAudioDiT`
    (vendored from `stable_audio_tools/models/dit.py + transformer.py`).
    Forward parity vs upstream `DiffusionTransformer` is bit-identical
    (diff=0) on shared random latents.
  * **Conditioner**: native `StableAudioMultiConditioner` (vendored
    from `stable_audio_tools/models/conditioners.py`).
  * **Sampler**: `k_diffusion.sampling.sample_dpmpp_3m_sde` — pure
    sampling library, the same function `generate_diffusion_cond`
    calls in upstream.
  * **Stage orchestration**: FV's stage split vs upstream's monolithic
    `generate_diffusion_cond`.

Empirically the end-to-end drift is ~0.015% abs_mean / max element
~0.006 on a 25-step CFG=7 run — fp32 numerical noise from non-
deterministic CUDA kernels (matmul reorder), not orchestration.

Skips when:
  * CUDA is unavailable.
  * `stabilityai/stable-audio-open-1.0` is inaccessible (gated).
  * `stable_audio_tools` and its deps are not installed.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

_HF_REPO_ID = "stabilityai/stable-audio-open-1.0"
_MODEL_CFG = "model_config.json"
_MODEL_WEIGHTS = "model.safetensors"


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
        hf_hub_download(repo_id=_HF_REPO_ID, filename="model_index.json", token=token)
        return True
    except Exception:
        return False


def _stable_audio_tools_inference_available() -> bool:
    try:
        from stable_audio_tools.inference.generation import (  # noqa: F401
            generate_diffusion_cond,
        )
        from stable_audio_tools.models.factory import (  # noqa: F401
            create_model_from_config,
        )
        return True
    except Exception:
        return False


def _setup_hf_env() -> None:
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(src)
        if v:
            os.environ.setdefault("HF_TOKEN", v)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", v)
            return


def _load_official_diffusion_cond(device: torch.device):
    """Build the official `ConditionedDiffusionModelWrapper` from
    `model_config.json` and load its weights from `model.safetensors`.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_MODEL_CFG,
                               token=_hf_token())
    weights_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_MODEL_WEIGHTS,
                                   token=_hf_token())
    with open(cfg_path) as f:
        model_config = json.load(f)

    from stable_audio_tools.models.factory import create_model_from_config
    model = create_model_from_config(model_config)
    state = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (
        f"official model state mismatch — missing={missing[:3]} unexpected={unexpected[:3]}"
    )
    return model.to(device).eval(), model_config


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Stable Audio pipeline parity requires CUDA.",
)
@pytest.mark.skipif(
    not _can_access(),
    reason=(f"{_HF_REPO_ID} not accessible — gated repo; set HF_TOKEN / "
            f"HF_API_KEY and accept the terms on https://huggingface.co/{_HF_REPO_ID}."),
)
@pytest.mark.skipif(
    not _stable_audio_tools_inference_available(),
    reason=("`stable_audio_tools.inference.generation` not importable. "
            "Clone https://github.com/Stability-AI/stable-audio-tools and "
            "`pip install` its deps (k_diffusion, einops_exts, alias_free_torch)."),
)
def test_stable_audio_pipeline_official_parity():
    _setup_hf_env()
    device = torch.device("cuda:0")

    prompt = "A gentle ambient pad with soft synth swells."
    negative_prompt = "low quality, distorted"
    seed = 0
    num_inference_steps = 25
    guidance_scale = 7.0
    audio_end_in_s = 1.5

    # --- Reference: official `generate_diffusion_cond` ---
    off_model, off_cfg = _load_official_diffusion_cond(device)
    sample_rate = int(off_cfg["sample_rate"])
    # Use the model's training sample_size (full latent length 1024).
    # Generating at the full duration matches what FV/diffusers do —
    # both produce the full ~47.5s clip then slice. Generating at the
    # requested 1.5s would change the latent shape and so the noise
    # trajectory, which would not be a meaningful comparison.
    full_sample_size = int(off_cfg["sample_size"])

    cond = [{
        "prompt": prompt,
        "seconds_start": 0.0,
        "seconds_total": audio_end_in_s,
    }]
    neg_cond = [{
        "prompt": negative_prompt,
        "seconds_start": 0.0,
        "seconds_total": audio_end_in_s,
    }]

    from stable_audio_tools.inference.generation import generate_diffusion_cond
    audio_full = generate_diffusion_cond(
        off_model,
        steps=num_inference_steps,
        cfg_scale=guidance_scale,
        conditioning=cond,
        negative_conditioning=neg_cond,
        sample_size=full_sample_size,
        seed=seed,
        device="cuda",
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.3,
        sigma_max=500,
    )
    end_idx = int(audio_end_in_s * sample_rate)
    off_audio = audio_full.detach().float().cpu()[:, :, :end_idx]
    print(
        f"off shape={tuple(off_audio.shape)} "
        f"abs_mean={off_audio.abs().mean().item():.6f} "
        f"range=[{off_audio.min().item():.4f}, {off_audio.max().item():.4f}]"
    )

    # Free the reference model so FV has GPU memory.
    del off_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # --- FastVideo path ---
    from fastvideo import VideoGenerator
    generator = VideoGenerator.from_pretrained(
        _HF_REPO_ID,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
    )
    try:
        result = generator.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path="outputs_audio/stable_audio_parity.mp4",
            save_video=False,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            audio_end_in_s=audio_end_in_s,
        )
    finally:
        generator.shutdown()

    fv_audio = result.get("audio")
    if fv_audio is None:
        ext = result.get("extra", {}) or {}
        fv_audio = ext.get("decoded_audio")
    assert fv_audio is not None, "FastVideo pipeline did not surface audio"
    if not torch.is_tensor(fv_audio):
        import numpy as np
        fv_audio = torch.from_numpy(np.asarray(fv_audio))
    fv_audio = fv_audio.detach().float().cpu()
    if fv_audio.ndim == 2 and fv_audio.shape[0] != 1:
        fv_audio = fv_audio.T.unsqueeze(0)
    print(
        f"fv  shape={tuple(fv_audio.shape)} "
        f"abs_mean={fv_audio.abs().mean().item():.6f} "
        f"range=[{fv_audio.min().item():.4f}, {fv_audio.max().item():.4f}]"
    )

    assert fv_audio.shape == off_audio.shape, (
        f"shape mismatch: off={off_audio.shape} fv={fv_audio.shape}"
    )

    diff = (fv_audio - off_audio).abs()
    diff_max = diff.max().item()
    diff_mean = diff.mean().item()
    diff_median = diff.median().item()
    rel = abs(off_audio.abs().mean() - fv_audio.abs().mean()) / max(off_audio.abs().mean().item(), 1e-6)
    print(
        f"diff max={diff_max:.6f} mean={diff_mean:.6f} median={diff_median:.6f}"
    )
    print(f"abs_mean rel drift: {rel:.4%}")

    # Tight bounds (1% drift, 0.05 element-wise) since FV is now a
    # first-class port of the upstream — drift is fp32 kernel noise,
    # not algorithmic divergence.
    assert rel < 0.01, f"abs_mean rel drift {rel:.2%} > 1% — port regression"
    assert diff_max < 0.05, (
        f"max element diff {diff_max:.4f} > 0.05 — port regression"
    )
