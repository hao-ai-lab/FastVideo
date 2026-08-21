# SPDX-License-Identifier: Apache-2.0
"""Wan VAE latent denormalization regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastvideo.mlx_runtime.wan_vae import (
    WanVAEConfigView,
    denormalize_latents_np,
)


def test_denormalize_matches_prompt_to_video_formula() -> None:
    """``latents / (1/std) + mean`` used by the hybrid script."""
    cfg = WanVAEConfigView(
        z_dim=2,
        latents_mean=(0.1, -0.2),
        latents_std=(2.0, 0.5),
    )
    lat = np.ones((1, 2, 1, 1, 1), dtype=np.float32)
    out = denormalize_latents_np(lat, cfg)
    # z * std + mean
    np.testing.assert_allclose(out[0, 0, 0, 0, 0], 1.0 * 2.0 + 0.1)
    np.testing.assert_allclose(out[0, 1, 0, 0, 0], 1.0 * 0.5 + (-0.2))


def test_full_vae_config_and_denormalize_formula() -> None:
    """Config load + denormalize formula vs the hybrid script's tensor math."""
    import torch

    snaps = list(
        Path.home()
        .joinpath(
            ".cache/huggingface/hub/"
            "models--FastVideo--FastWan2.2-TI2V-5B-FullAttn-Diffusers/"
            "snapshots"
        )
        .glob("*/vae")
    )
    if not snaps:
        pytest.skip("Wan2.2 VAE not in HF cache")
    vae_dir = snaps[0]
    cfg = WanVAEConfigView.from_vae_dir(vae_dir)
    assert cfg.z_dim == 48
    rng = np.random.default_rng(1)
    lat = (rng.standard_normal((1, 48, 3, 8, 8)) * 0.3).astype(np.float32)
    lat_dn = denormalize_latents_np(lat, cfg)
    mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).view(
        1, -1, 1, 1, 1
    )
    inv_std = (1.0 / torch.tensor(cfg.latents_std, dtype=torch.float32)).view(
        1, -1, 1, 1, 1
    )
    lat_t = torch.from_numpy(lat)
    ref = (lat_t / inv_std + mean).numpy()
    np.testing.assert_allclose(lat_dn, ref, atol=1e-5, rtol=1e-5)
