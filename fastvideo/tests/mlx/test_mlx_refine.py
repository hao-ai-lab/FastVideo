# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the MLX two-pass refine pipeline.

Mirrors the LTX-2 refine stage math (init resolution split, spatial
upsample, Gaussian re-noise) without requiring Metal / a real DiT. The
orchestration helpers that need ``mx`` are exercised only when MLX is
importable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from examples.inference.basic.mlx_wan22_generate import DEFAULT_HEIGHT, DEFAULT_NUM_FRAMES, DEFAULT_WIDTH
from fastvideo.mlx_runtime.refine import (
    DEFAULT_REFINE_SIGMA,
    default_refine_timesteps,
    plan_refine_resolutions,
    prepare_refine_latents,
    refine_sigma_from_schedule,
    upsample_latents_spatial,
)
from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, add_noise


def _fastwan_schedule() -> MLXDMDSchedule:
    """FastWan's flow-match schedule: ``sigma == timestep / 1000``."""
    timesteps = np.arange(1000, 0, -1, dtype=np.float64)
    return MLXDMDSchedule(sigmas=timesteps / 1000.0, timesteps=timesteps)


def test_plan_refine_resolutions_splits_even_target() -> None:
    plan = plan_refine_resolutions(
        height=480,
        width=832,
        num_frames=81,
        spatial_scale=2,
        vae_spatial_compression=8,
        vae_temporal_compression=4,
        patch_size=(1, 2, 2),
    )
    assert plan.stage1_height == 240
    assert plan.stage1_width == 416
    assert plan.target_height == 480
    assert plan.target_width == 832
    assert plan.stage1_latent_height == 30
    assert plan.stage1_latent_width == 52
    assert plan.stage2_latent_height == 60
    assert plan.stage2_latent_width == 104
    assert plan.latent_frames == 21
    assert plan.spatial_scale == 2


def test_plan_refine_disabled_collapses_to_single_pass() -> None:
    plan = plan_refine_resolutions(
        height=480,
        width=832,
        num_frames=81,
        spatial_scale=2,
        enabled=False,
    )
    assert plan.spatial_scale == 1
    assert plan.stage1_height == plan.target_height == 480
    assert plan.stage1_width == plan.target_width == 832


@pytest.mark.parametrize(
    ("height", "width", "num_frames", "message"),
    [(481, 832, 81, "height/width"), (480, 833, 81, "height/width"), (480, 832, 82, "num_frames")],
)
def test_plan_refine_rejects_requests_the_vae_would_truncate(
    height: int,
    width: int,
    num_frames: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_refine_resolutions(height=height, width=width, num_frames=num_frames, enabled=False)


def test_plan_refine_rejects_odd_target() -> None:
    with pytest.raises(ValueError, match="divisible by"):
        plan_refine_resolutions(height=481, width=832, num_frames=81, spatial_scale=2)


def test_plan_refine_rejects_vae_misaligned_stage1() -> None:
    # 500/2 = 250, 250 % 8 != 0 → rejected (same guard as LTX2RefineInitStage).
    with pytest.raises(ValueError, match="divisible by"):
        plan_refine_resolutions(
            height=500,
            width=800,
            num_frames=81,
            spatial_scale=2,
            vae_spatial_compression=8,
        )


def test_plan_refine_wan22_5b_720p_geometry() -> None:
    # Wan2.2 TI2V-5B native 704x1280 with 16x spatial VAE compression.
    plan = plan_refine_resolutions(
        height=704,
        width=1280,
        num_frames=121,
        spatial_scale=2,
        vae_spatial_compression=16,
        vae_temporal_compression=4,
        patch_size=(1, 2, 2),
    )
    assert plan.stage1_height == 352
    assert plan.stage1_width == 640
    assert plan.stage1_latent_height == 22
    assert plan.stage1_latent_width == 40
    assert plan.stage2_latent_height == 44
    assert plan.stage2_latent_width == 80
    assert plan.latent_frames == 31


def test_plan_refine_wan22_default_geometry() -> None:
    plan = plan_refine_resolutions(
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        num_frames=DEFAULT_NUM_FRAMES,
        spatial_scale=2,
        vae_spatial_compression=16,
        vae_temporal_compression=4,
        patch_size=(1, 2, 2),
    )
    assert (plan.stage1_latent_height, plan.stage1_latent_width) == (14, 26)
    assert (plan.stage2_latent_height, plan.stage2_latent_width) == (28, 52)
    assert plan.latent_frames == 31


def test_upsample_nearest_repeats_spatial_samples() -> None:
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((1, 4, 2, 3, 5)).astype(np.float32)
    up = upsample_latents_spatial(latents, scale=2, mode="nearest")
    assert up.shape == (1, 4, 2, 6, 10)
    # Every 2x2 block equals the source sample broadcast across the block.
    for y in range(3):
        for x in range(5):
            block = up[0, :, :, 2 * y:2 * y + 2, 2 * x:2 * x + 2]
            src = latents[0, :, :, y, x][:, :, None, None]
            np.testing.assert_array_equal(block, np.broadcast_to(src, block.shape))


def test_upsample_bilinear_preserves_constants() -> None:
    latents = np.full((1, 2, 1, 4, 4), 3.5, dtype=np.float32)
    up = upsample_latents_spatial(latents, scale=2, mode="bilinear")
    assert up.shape == (1, 2, 1, 8, 8)
    np.testing.assert_allclose(up, 3.5, rtol=0, atol=1e-6)


def test_upsample_scale_one_is_identity() -> None:
    rng = np.random.default_rng(1)
    latents = rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float32)
    out = upsample_latents_spatial(latents, scale=1)
    np.testing.assert_array_equal(out, latents)


def test_prepare_refine_latents_matches_add_noise_formula() -> None:
    rng = np.random.default_rng(2)
    clean = rng.standard_normal((1, 4, 2, 3, 5)).astype(np.float32)
    noise = rng.standard_normal((1, 4, 2, 6, 10)).astype(np.float32)
    sigma = 0.75

    got = prepare_refine_latents(
        clean,
        scale=2,
        sigma=sigma,
        noise=noise,
        add_noise_flag=True,
        upsample_mode="nearest",
    )
    up = upsample_latents_spatial(clean, scale=2, mode="nearest")
    expected = add_noise(up, noise, sigma)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_prepare_refine_latents_no_noise_returns_upsample() -> None:
    rng = np.random.default_rng(3)
    clean = rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float32)
    got = prepare_refine_latents(clean, scale=2, add_noise_flag=False, upsample_mode="nearest")
    expected = upsample_latents_spatial(clean, scale=2, mode="nearest")
    np.testing.assert_array_equal(got, expected)


def test_prepare_refine_latents_seed_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    clean = rng.standard_normal((1, 2, 1, 3, 3)).astype(np.float32)
    a = prepare_refine_latents(clean, scale=2, sigma=0.5, seed=123, upsample_mode="nearest")
    b = prepare_refine_latents(clean, scale=2, sigma=0.5, seed=123, upsample_mode="nearest")
    np.testing.assert_array_equal(a, b)


def test_refine_sigma_from_schedule_uses_first_timestep() -> None:
    timesteps = np.arange(1000, dtype=np.float64)
    sigmas = np.linspace(1.0, 1e-3, 1000, dtype=np.float64)
    schedule = MLXDMDSchedule(sigmas=sigmas, timesteps=timesteps)
    # First refine timestep drives the stage-2 start sigma.
    assert refine_sigma_from_schedule(schedule, [999, 757, 522]) == pytest.approx(float(sigmas[999]))
    assert refine_sigma_from_schedule(schedule, [522]) == pytest.approx(float(sigmas[522]))


def test_default_refine_sigma_matches_ltx2_stage2_head() -> None:
    # Documented contract: DEFAULT_REFINE_SIGMA tracks LTX-2's first
    # STAGE_2_DISTILLED_SIGMA_VALUES entry so the fallback is intentional.
    assert math.isclose(DEFAULT_REFINE_SIGMA, 0.909375, rel_tol=0, abs_tol=0)


def test_full_noise_handoff_would_discard_stage1() -> None:
    """Why the stage-2 grid may not open at the stage-1 timestep.

    FastWan's DMD grid starts at ``t=1000``, i.e. ``sigma == 1``. The hand-off
    is ``(1 - sigma) * upsampled + sigma * noise``, so starting stage 2 there
    weights the stage-1 result at exactly zero: refine silently degrades to a
    plain full-resolution run costing two passes.
    """
    schedule = _fastwan_schedule()
    assert refine_sigma_from_schedule(schedule, [1000, 757, 522]) == pytest.approx(1.0)

    clean = np.random.default_rng(0).standard_normal((1, 2, 1, 4, 4)).astype(np.float32)
    handoff = prepare_refine_latents(clean, scale=2, sigma=1.0, seed=7, upsample_mode="nearest")
    upsampled = upsample_latents_spatial(clean, scale=2, mode="nearest")
    # Nothing of stage 1 survives.
    assert not np.allclose(handoff, upsampled)


def test_default_refine_timesteps_drops_the_full_noise_step() -> None:
    schedule = _fastwan_schedule()
    steps = default_refine_timesteps(schedule, [1000, 757, 522])
    assert steps == [757.0, 522.0]
    # The resulting hand-off keeps a real share of stage 1.
    sigma = refine_sigma_from_schedule(schedule, steps)
    assert 0.0 < sigma < 1.0


def test_default_refine_timesteps_keeps_an_already_valid_grid() -> None:
    schedule = _fastwan_schedule()
    assert default_refine_timesteps(schedule, [757, 522]) == [757.0, 522.0]


def test_default_refine_timesteps_rejects_an_all_full_noise_grid() -> None:
    schedule = _fastwan_schedule()
    with pytest.raises(ValueError, match="No usable refine timesteps"):
        default_refine_timesteps(schedule, [1000])
