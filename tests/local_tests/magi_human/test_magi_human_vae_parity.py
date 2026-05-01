# SPDX-License-Identifier: Apache-2.0
"""Parity test: FastVideo `AutoencoderKLWan` vs upstream `Wan2_2_VAE`.

MagiHuman uses the Wan 2.2 TI2V-5B VAE. The two implementations
compared here are:

  * Upstream (SandAI port) — `inference/model/vae2_2/vae2_2_module.py::Wan2_2_VAE`
    loaded from `Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` (the official .pth
    inside the daVinci-MagiHuman repo). This is the reference.
  * FastVideo — `fastvideo.models.vaes.wanvae.AutoencoderKLWan` (the
    class registered as `EntryClass` and resolved by the VAE component
    loader at runtime; this is what `MagiHumanBaseConfig.vae_config`
    materializes when the magi pipeline runs). Weights are loaded from
    a Diffusers-format `vae/` subdir (`config.json` +
    `diffusion_pytorch_model.safetensors`).

This test decodes the same random latent through both and asserts the
decoded videos are close. Catches regressions in:
  - FastVideo's `AutoencoderKLWan` weight load / scale / shift handling.
  - Any deviation in `latents_mean` / `latents_std` baked into the
    Diffusers-format config vs the upstream constants.

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
    # FastVideo's `AutoencoderKLWan.decode(z)` expects the input to
    # ALREADY be in "decoder-input space" (the normalization is the
    # caller's job — `DecodingStage` applies it). So we mirror the
    # upstream transform here before calling decode.
    import glob

    from safetensors.torch import load_file as safetensors_load_file

    from fastvideo.configs.models.vaes import WanVAEConfig
    from fastvideo.models.loader.component_loader import get_diffusers_config
    from fastvideo.models.vaes.wanvae import AutoencoderKLWan

    diffusers_cfg = get_diffusers_config(model=str(fv_vae_dir))
    diffusers_cfg.pop("_class_name", None)
    diffusers_cfg.pop("_name_or_path", None)
    fv_config = WanVAEConfig()
    fv_config.load_encoder = False
    fv_config.load_decoder = True
    fv_config.update_model_arch(diffusers_cfg)
    fv_vae = AutoencoderKLWan(fv_config).to(device=device, dtype=torch.float32)

    # Mirror the VAE component loader: glob `*.safetensors`, merge, load
    # non-strictly so any unused buffers (per_channel_statistics, etc.)
    # don't fail the load.
    sf_files = glob.glob(os.path.join(str(fv_vae_dir), "*.safetensors"))
    assert sf_files, f"No safetensors files in {fv_vae_dir}"
    state = {}
    for sf in sf_files:
        state.update(safetensors_load_file(sf))
    fv_vae.load_state_dict(state, strict=False)
    fv_vae.eval()

    # Upstream's inner `_video_vae.decode(z, scale)` (line 874-877 of
    # inference/model/vae2_2/vae2_2_module.py) does:
    #     z = z / scale[1] + scale[0]    # where scale = [mean, 1/std]
    #     = z * std + mean
    # FastVideo's `AutoencoderKLWan.decode` expects the pre-denormalized
    # latent — apply the same transform externally to feed both paths
    # equivalently.
    latents_mean = torch.tensor(
        fv_config.arch_config.latents_mean, dtype=torch.float32, device=device,
    )
    latents_std = torch.tensor(
        fv_config.arch_config.latents_std, dtype=torch.float32, device=device,
    )
    z_denormalized = z * latents_std.view(1, -1, 1, 1, 1) + latents_mean.view(1, -1, 1, 1, 1)
    with torch.inference_mode():
        fv_out_tensor = fv_vae.decode(z_denormalized)
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

    # Wan VAE has a known fp32 op-ordering drift of ~8e-4 caused by
    # `z * std + mean` (FV) vs `z / (1/std) + mean` (upstream) at decode
    # normalization. This is a SHARED Wan-family bug, not magi-specific.
    # Tracked as OQ-7 in tests/local_tests/magi-human.md; tighten to
    # atol=1e-4 once the Wan VAE op-order fix lands.
    assert up_s.shape == fv_s.shape, (
        f"shape mismatch: up={up_s.shape} fv={fv_s.shape}"
    )
    diff = (up_s - fv_s).abs()
    print(
        f"diff max={diff.max().item():.6f} mean={diff.mean().item():.6f}"
    )
    assert_close(fv_s, up_s, atol=1e-3, rtol=1e-3)
