# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for spatial fast mode + flag composition."""

from __future__ import annotations

import numpy as np
import pytest

from fastvideo.mlx_runtime.fast_spatial import (
    apply_fast_spatial_upsample,
    plan_fast_spatial,
    resolve_spatial_mode,
)
from fastvideo.mlx_runtime.refine import upsample_latents_spatial


def test_resolve_spatial_mode_priority() -> None:
    assert resolve_spatial_mode(refine=False, fast_spatial=False) == "off"
    assert resolve_spatial_mode(refine=False, fast_spatial=True) == "fast_spatial"
    # Refine is the quality path and wins when both are requested.
    assert resolve_spatial_mode(refine=True, fast_spatial=True) == "refine"
    assert resolve_spatial_mode(refine=True, fast_spatial=False) == "refine"


def test_plan_fast_spatial_matches_refine_geometry() -> None:
    spatial = plan_fast_spatial(
        height=480,
        width=832,
        num_frames=81,
        spatial_scale=2,
        vae_spatial_compression=8,
    )
    assert spatial.enabled
    assert spatial.stage1_height == 240
    assert spatial.stage1_width == 416
    assert spatial.target_height == 480
    assert spatial.target_width == 832
    assert spatial.plan.stage1_latent_height == 30
    assert spatial.plan.stage2_latent_width == 104


def test_plan_fast_spatial_disabled() -> None:
    spatial = plan_fast_spatial(height=480, width=832, num_frames=81, enabled=False)
    assert not spatial.enabled
    assert spatial.scale == 1


def test_apply_fast_spatial_upsample_nearest() -> None:
    rng = np.random.default_rng(0)
    # stage1 latent 3x5 → scale 2 → 6x10
    clean = rng.standard_normal((1, 4, 2, 3, 5)).astype(np.float32)
    spatial = plan_fast_spatial(
        height=48,  # 48/8=6 stage2 latent h; stage1=24 → latent 3
        width=80,  # 80/8=10 stage2; stage1=40 → latent 5
        num_frames=9,
        spatial_scale=2,
        vae_spatial_compression=8,
        vae_temporal_compression=4,
        patch_size=(1, 1, 1),  # avoid patch alignment on tiny grid
        upsample_mode="nearest",
    )
    assert spatial.plan.stage1_latent_height == 3
    assert spatial.plan.stage2_latent_width == 10
    up = apply_fast_spatial_upsample(clean, spatial)
    expected = upsample_latents_spatial(clean, scale=2, mode="nearest")
    np.testing.assert_array_equal(up, expected)


def test_apply_fast_spatial_noop_when_disabled() -> None:
    rng = np.random.default_rng(1)
    clean = rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float32)
    spatial = plan_fast_spatial(height=32, width=32, num_frames=5, enabled=False, patch_size=(1, 1, 1))
    out = apply_fast_spatial_upsample(clean, spatial)
    np.testing.assert_array_equal(out, clean)


def test_plan_fast_spatial_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="upsample mode"):
        plan_fast_spatial(height=480, width=832, num_frames=81, upsample_mode="bicubic")
