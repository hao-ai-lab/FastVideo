# SPDX-License-Identifier: Apache-2.0
"""Parity test: FastVideo `AutoencoderKLWan` vs upstream `Wan2_2_VAE`.

MagiHuman uses the Wan 2.2 TI2V-5B VAE. Two Python implementations exist:

  * Upstream (SandAI port) — `inference/model/vae2_2/vae2_2_module.py::Wan2_2_VAE`
    loaded from `Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` (the official .pth).
  * FastVideo — `AutoencoderKLWan` (Diffusers format) loaded from
    `Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae/` (also first-party, Wan-AI's own
    Diffusers port).

This test decodes the same random latent through both and asserts the
decoded videos are close. Catches regressions in:
  - The Diffusers weight conversion Wan-AI shipped.
  - FastVideo's `AutoencoderKLWan` load / scale / shift handling.
  - Any deviation in `latents_mean` / `latents_std` between the two.

Skips when:
  - CUDA is unavailable.
  - The .pth is not locally available (requires ~2.8 GB download).
  - The converted MagiHuman Diffusers repo (or any `Wan-AI/*-Diffusers`
    repo with a `vae/` subdir) is not available locally.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VAE parity test requires CUDA.",
)
def test_magi_human_vae_decode_parity():
    repo_root = Path(__file__).resolve().parents[3]
    upstream_src = repo_root / "daVinci-MagiHuman"
    if not upstream_src.exists():
        pytest.skip(
            "Upstream daVinci-MagiHuman/ clone missing — no Wan2_2_VAE source."
        )

    fv_vae_dir = Path(os.getenv(
        "MAGI_HUMAN_VAE_DIR",
        repo_root / "converted_weights" / "magi_human_base" / "vae",
    ))
    if not (fv_vae_dir / "config.json").is_file():
        pytest.skip(f"FastVideo VAE dir missing at {fv_vae_dir}")

    # Upstream Wan2_2_VAE needs the raw .pth shipped by Wan-AI/Wan2.2-TI2V-5B
    # (NOT the -Diffusers variant; that one has safetensors, not .pth).
    try:
        from huggingface_hub import hf_hub_download
        pth_path = hf_hub_download(
            repo_id="Wan-AI/Wan2.2-TI2V-5B", filename="Wan2.2_VAE.pth",
        )
    except Exception as exc:
        pytest.skip(f"Wan2.2_VAE.pth not available: {exc}")

    # Push upstream + install compiler stubs (the VAE module itself doesn't
    # need magi_compiler, but `inference.*` imports pull in siblings that do).
    from tests.local_tests.helpers.magi_human_upstream import install_stubs
    install_stubs()

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    # Tiny latent so the test stays well inside GPU memory budget.
    # z_dim=48, T=1, H=4, W=4 -> VAE decodes to [1, 3, 1 (or 1+4*0), 64, 64]
    z = torch.randn((1, 48, 1, 4, 4), dtype=torch.float32, device=device)

    # --- Upstream decode ---
    from inference.model.vae2_2 import Wan2_2_VAE
    up_vae = Wan2_2_VAE(
        vae_pth=pth_path,
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        # Wan2_2_VAE.decode expects a (C, T, H, W) latent (no batch dim);
        # see inference/pipeline/video_generate.py:494 — `self.vae.decode(latent.squeeze(0).to(self.dtype), ...)`.
        up_out = up_vae.decode(z[0]).detach().float().cpu()

    del up_vae
    import gc; gc.collect(); torch.cuda.empty_cache()

    # --- FastVideo decode ---
    # Upstream `Wan2_2_VAE.decode(z)` internally normalizes via
    # `(z - latents_mean) / latents_std` before feeding the decoder
    # (see `scale = [mean, 1.0/std]` and the _video_vae.decode call).
    # Diffusers' `AutoencoderKLWan.decode(z)` expects the input to ALREADY
    # be in "decoder-input space" (the normalization is the caller's job
    # in FastVideo's DecodingStage). Apply the same normalization here.
    from diffusers import AutoencoderKLWan as DiffusersAutoencoderKLWan
    fv_vae = DiffusersAutoencoderKLWan.from_pretrained(
        str(fv_vae_dir), torch_dtype=torch.float32,
    ).to(device)
    fv_vae.eval()
    # Upstream's inner `_video_vae.decode(z, scale)` (line 874-877 of
    # inference/model/vae2_2/vae2_2_module.py) does:
    #     z = z / scale[1] + scale[0]    # where scale = [mean, 1/std]
    #     = z * std + mean
    # So upstream's caller passes z in "normalized diffusion space" and
    # the VAE denormalizes internally.  Diffusers' `AutoencoderKLWan.decode`
    # expects the pre-denormalized latent — apply the same transform
    # externally to feed both paths equivalently.
    latents_mean = torch.tensor(fv_vae.config.latents_mean, dtype=torch.float32, device=device)
    latents_std = torch.tensor(fv_vae.config.latents_std, dtype=torch.float32, device=device)
    z_denormalized = z * latents_std.view(1, -1, 1, 1, 1) + latents_mean.view(1, -1, 1, 1, 1)
    with torch.inference_mode():
        fv_out_tensor = fv_vae.decode(z_denormalized, return_dict=False)[0]
        fv_out = fv_out_tensor.detach().float().cpu()

    # Both sides should return a video tensor of shape [..., C, T_dec, H_dec, W_dec].
    # Normalize shapes for comparison — upstream returns a list per-video or a
    # single tensor depending on CP group; we just squeeze batch dims.
    def _squeeze(t):
        while t.ndim > 4 and t.shape[0] == 1:
            t = t[0]
        return t

    up_s = _squeeze(up_out)
    fv_s = _squeeze(fv_out)
    print(
        f"up shape={tuple(up_s.shape)} abs_mean={up_s.abs().mean().item():.4f} "
        f"range=[{up_s.min().item():.4f}, {up_s.max().item():.4f}]"
    )
    print(
        f"fv shape={tuple(fv_s.shape)} abs_mean={fv_s.abs().mean().item():.4f} "
        f"range=[{fv_s.min().item():.4f}, {fv_s.max().item():.4f}]"
    )

    # The two implementations are from the same team (Wan-AI) and should
    # agree to within fp32 numerical noise. Use a loose tolerance since
    # the pipelines may differ slightly in normalization ordering.
    assert up_s.shape == fv_s.shape, (
        f"shape mismatch: up={up_s.shape} fv={fv_s.shape}"
    )
    diff = (up_s - fv_s).abs()
    print(
        f"diff max={diff.max().item():.6f} mean={diff.mean().item():.6f}"
    )
    assert_close(fv_s, up_s, atol=5e-2, rtol=5e-2)
