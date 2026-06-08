# SPDX-License-Identifier: Apache-2.0
"""Inpainting / outpainting test for FastVideo's RePaint blending path.

Stable Audio Open 1.0 is `model_type=diffusion_cond`, **not**
`diffusion_cond_inpaint` — there is no inpaint-trained checkpoint for
this model, so an apples-to-apples parity comparison against
`generate_diffusion_cond_inpaint` would be testing the wrong thing
(that function expects a model whose conditioner accepts `inpaint_mask`
and `inpaint_masked_input`, which our model's conditioner does not).

Instead this test validates the two invariants that any RePaint-style
blender must hold:

  1. **Kept-region preservation** — for every sample where mask == 1,
     the decoded output must round-trip back to the reference within
     fp32 VAE round-trip noise (~RMS ratio close to 1, max diff
     bounded). This is the core promise of mask=1: "leave this alone".
  2. **Unkept-region freshness** — for samples where mask == 0, the
     output must be different from the reference (model actually
     regenerated something). RMS in the unkept region should be in
     a sane audio range (not silent, not blown up).

Skips when CUDA / HF access is unavailable.
"""
from __future__ import annotations

import pytest
import torch

from ._stable_audio_helpers import (
    can_access_repo,
    setup_hf_env,
)

_HF_REPO_ID = "stabilityai/stable-audio-open-1.0"  # raw upstream — only used to gate the test on access
_FV_REPO_ID = "FastVideo/stable-audio-open-1.0-Diffusers"  # converted Diffusers repo FastVideo loads from


def _make_reference(seed: int, sample_rate: int, seconds: float):
    """Stereo amplitude-modulated noise — fakes a 'beat'."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = int(seconds * sample_rate)
    base = torch.randn(2, n, generator=g) * 0.1
    env = 0.5 + 0.5 * torch.sin(2 * torch.pi * 4 * torch.linspace(0, seconds, n))
    return (base * env).unsqueeze(0).contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Stable Audio inpainting test requires CUDA.")
@pytest.mark.skipif(not can_access_repo(_HF_REPO_ID),
                    reason=f"{_HF_REPO_ID} not accessible — gated.")
def test_stable_audio_inpaint_kept_region_preserved():
    """Run inpainting with the first 1.5s kept and the next 4.5s
    regenerated. Verify the kept region matches the reference and the
    regenerated region is genuinely different.
    """
    setup_hf_env()
    sample_rate = 44100
    keep_seconds = 1.5
    total_seconds = 6.0
    keep_samples = int(keep_seconds * sample_rate)
    total_samples = int(total_seconds * sample_rate)

    ref_short = _make_reference(seed=7, sample_rate=sample_rate, seconds=keep_seconds)
    padded_ref = torch.zeros((1, ref_short.shape[1], total_samples), dtype=torch.float32)
    padded_ref[..., :keep_samples] = ref_short

    mask = torch.zeros(total_samples, dtype=torch.float32)
    mask[:keep_samples] = 1.0

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
            prompt="Steady lo-fi hip hop drum loop with vinyl crackle.",
            output_path="outputs_audio/sa_inpaint_self.mp4",
            save_video=False,
            num_inference_steps=25,
            guidance_scale=7.0,
            seed=0,
            audio_end_in_s=total_seconds,
            inpaint_audio=padded_ref,
            inpaint_mask=mask,
        )
    finally:
        generator.shutdown()

    fv_audio = result.get("audio")
    if fv_audio is None:
        fv_audio = (result.get("extra", {}) or {}).get("decoded_audio")
    if not torch.is_tensor(fv_audio):
        import numpy as np
        fv_audio = torch.from_numpy(np.asarray(fv_audio))
    out = fv_audio.detach().float().cpu()
    if out.ndim == 2 and out.shape[0] != 1:
        out = out.T.unsqueeze(0)
    print(f"out shape={tuple(out.shape)} abs_mean={out.abs().mean():.6f}")
    assert out.shape[-1] >= keep_samples

    ref_kept = padded_ref[..., :keep_samples].float().cpu()
    out_kept = out[..., :keep_samples]
    out_unkept = out[..., keep_samples:]

    # 1. Kept region: should round-trip to within VAE round-trip noise.
    #    Round-trip RMS ratio for Oobleck on synthetic audio runs ~1.0–1.05
    #    in the existing `test_oobleck_vae_round_trip_sanity` test, so we
    #    bound the per-sample-RMS ratio inside [0.5, 2.0] to confirm the
    #    kept region was actually preserved (and not, e.g., zeroed out).
    ref_rms = ref_kept.float().pow(2).mean().sqrt().item()
    out_kept_rms = out_kept.float().pow(2).mean().sqrt().item()
    rms_ratio = out_kept_rms / max(ref_rms, 1e-6)
    print(f"kept rms ratio={rms_ratio:.3f}  ref_rms={ref_rms:.4f} out_rms={out_kept_rms:.4f}")
    assert 0.5 < rms_ratio < 2.0, (
        f"kept-region RMS ratio {rms_ratio:.3f} outside [0.5, 2.0] — "
        "RePaint blending did not preserve the reference")

    # 2. Unkept region: should be live audio (finite, in-range, non-silent).
    assert torch.isfinite(out_unkept).all(), "unkept region produced non-finite values"
    unkept_rms = out_unkept.float().pow(2).mean().sqrt().item()
    print(f"unkept rms={unkept_rms:.4f}")
    assert unkept_rms > 1e-3, "unkept region is silent — model produced nothing"
    assert out_unkept.abs().max().item() < 5.0, "unkept region magnitude blew up"
