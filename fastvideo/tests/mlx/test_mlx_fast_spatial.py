# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for spatial fast mode + flag composition."""

from __future__ import annotations

import numpy as np
import pytest

from examples.inference.basic.mlx_wan22_generate import DEFAULT_HEIGHT, DEFAULT_NUM_FRAMES, DEFAULT_WIDTH
from fastvideo.mlx_runtime.fast_spatial import (
    DEFAULT_FAST_SPATIAL_SHARPEN,
    apply_fast_spatial_upsample,
    plan_fast_spatial,
    resolve_spatial_mode,
)
from fastvideo.mlx_runtime.frame_upsample import upsample_frames


def _frames(count: int, height: int, width: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(count)]


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
    assert spatial.sharpen == DEFAULT_FAST_SPATIAL_SHARPEN


def test_plan_fast_spatial_disabled() -> None:
    spatial = plan_fast_spatial(height=480, width=832, num_frames=81, enabled=False)
    assert not spatial.enabled
    assert spatial.scale == 1


def test_plan_fast_spatial_accepts_wan22_defaults() -> None:
    assert (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_NUM_FRAMES) == (448, 832, 121)
    spatial = plan_fast_spatial(
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        num_frames=DEFAULT_NUM_FRAMES,
        spatial_scale=2,
        vae_spatial_compression=16,
        vae_temporal_compression=4,
        patch_size=(1, 2, 2),
    )
    assert (spatial.plan.stage1_latent_height, spatial.plan.stage1_latent_width) == (14, 26)
    assert (spatial.plan.stage2_latent_height, spatial.plan.stage2_latent_width) == (28, 52)


def test_apply_fast_spatial_upsample_resizes_decoded_frames() -> None:
    """The upsample is pixel-space: it takes decoded frames, not latents.

    Interpolating latents instead is what made spatial fast mode incoherent —
    a blended Wan latent is off the decoder's manifold, so the decode came back
    as the right silhouette under a smeared veil.
    """
    spatial = plan_fast_spatial(
        height=480,
        width=832,
        num_frames=81,
        spatial_scale=2,
        vae_spatial_compression=8,
        upsample_mode="lanczos",
        sharpen=0.0,
    )
    frames = _frames(3, 240, 416)
    out = apply_fast_spatial_upsample(frames, spatial)

    assert len(out) == 3
    assert all(frame.shape == (480, 832, 3) for frame in out)
    assert all(frame.dtype == np.uint8 for frame in out)
    expected = upsample_frames(frames, width=832, height=480, mode="lanczos", sharpen=0.0)
    for got, want in zip(out, expected, strict=True):
        np.testing.assert_array_equal(got, want)
    # Inputs are not mutated.
    assert all(frame.shape == (240, 416, 3) for frame in frames)


def test_apply_fast_spatial_upsample_applies_sharpen() -> None:
    frames = _frames(1, 240, 416)
    soft = plan_fast_spatial(height=480, width=832, num_frames=81, spatial_scale=2, sharpen=0.0)
    crisp = plan_fast_spatial(height=480, width=832, num_frames=81, spatial_scale=2, sharpen=0.8)
    assert not np.array_equal(
        apply_fast_spatial_upsample(frames, soft)[0],
        apply_fast_spatial_upsample(frames, crisp)[0],
    )


def test_apply_fast_spatial_noop_when_disabled() -> None:
    frames = _frames(2, 32, 32, seed=1)
    spatial = plan_fast_spatial(height=32, width=32, num_frames=5, enabled=False, patch_size=(1, 1, 1))
    out = apply_fast_spatial_upsample(frames, spatial)
    for got, want in zip(out, frames, strict=True):
        np.testing.assert_array_equal(got, want)


def test_plan_fast_spatial_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="upsample mode"):
        plan_fast_spatial(height=480, width=832, num_frames=81, upsample_mode="bicubic")


def test_plan_fast_spatial_rejects_negative_sharpen() -> None:
    with pytest.raises(ValueError, match="sharpen"):
        plan_fast_spatial(height=480, width=832, num_frames=81, sharpen=-0.1)


def test_geometry_errors_name_fast_spatial_not_refine() -> None:
    """Both modes share the resolution splitter; the error must name the caller.

    832 is not divisible by 3, so scale 3 is rejected at plan time.
    """
    with pytest.raises(ValueError, match=r"^fast-spatial requires"):
        plan_fast_spatial(height=480, width=832, num_frames=81, spatial_scale=3)
