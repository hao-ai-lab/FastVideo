# SPDX-License-Identifier: Apache-2.0
"""CPU-only correctness contract for the on-device MLX DMD sampler.

The sampler's arithmetic uses only ``+``/``-``/``*``, so NumPy arrays are a
valid stand-in for MLX arrays here: these tests validate the *math* against the
torch reference formulas without requiring an MLX/Metal device.
"""

import numpy as np

from fastvideo.mlx_runtime.sampling import (
    MLXDMDSchedule,
    add_noise,
    dmd_step,
    pred_noise_to_pred_video,
)


def _reference_schedule() -> MLXDMDSchedule:
    # A small monotonic flow-match-like schedule: timesteps 0..999, sigmas in
    # (0, 1]. Exact values are irrelevant; the lookup must pick the nearest.
    timesteps = np.arange(1000, dtype=np.float64)
    sigmas = np.linspace(1.0, 1e-3, 1000, dtype=np.float64)
    return MLXDMDSchedule(sigmas=sigmas, timesteps=timesteps)


def test_sigma_for_picks_nearest_timestep() -> None:
    schedule = _reference_schedule()
    # Exact hits.
    assert schedule.sigma_for(0) == schedule.sigmas[0]
    assert schedule.sigma_for(999) == schedule.sigmas[999]
    # Nearest-neighbour rounding for a value between grid points.
    assert schedule.sigma_for(522.4) == schedule.sigmas[522]
    assert schedule.sigma_for(521.6) == schedule.sigmas[522]


def test_pred_noise_to_pred_video_matches_reference_formula() -> None:
    rng = np.random.default_rng(0)
    pred_noise = rng.standard_normal((1, 16, 3, 8, 8)).astype(np.float32)
    noise_input = rng.standard_normal((1, 16, 3, 8, 8)).astype(np.float32)
    sigma = 0.37

    got = pred_noise_to_pred_video(pred_noise, noise_input, sigma)
    expected = noise_input - sigma * pred_noise
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_add_noise_matches_reference_formula() -> None:
    rng = np.random.default_rng(1)
    clean = rng.standard_normal((1, 16, 3, 8, 8)).astype(np.float32)
    noise = rng.standard_normal((1, 16, 3, 8, 8)).astype(np.float32)
    sigma = 0.62

    got = add_noise(clean, noise, sigma)
    expected = (1.0 - sigma) * clean + sigma * noise
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_dmd_step_final_returns_clean_prediction() -> None:
    schedule = _reference_schedule()
    rng = np.random.default_rng(2)
    pred_noise = rng.standard_normal((1, 16, 3, 8, 8)).astype(np.float32)
    noise_input = rng.standard_normal((1, 16, 3, 8, 8)).astype(np.float32)

    out = dmd_step(
        latents=noise_input,
        noise_input_latent=noise_input,
        pred_noise=pred_noise,
        schedule=schedule,
        timestep=522,
        next_timestep=None,
    )
    sigma = schedule.sigma_for(522)
    np.testing.assert_allclose(out, noise_input - sigma * pred_noise, rtol=0, atol=0)


def test_dmd_step_intermediate_renoises_to_next_level() -> None:
    schedule = _reference_schedule()
    rng = np.random.default_rng(3)
    pred_noise = rng.standard_normal((1, 16, 3, 8, 8)).astype(np.float32)
    noise_input = rng.standard_normal((1, 16, 3, 8, 8)).astype(np.float32)
    renoise = rng.standard_normal((1, 16, 3, 8, 8)).astype(np.float32)

    out = dmd_step(
        latents=noise_input,
        noise_input_latent=noise_input,
        pred_noise=pred_noise,
        schedule=schedule,
        timestep=1000,
        next_timestep=757,
        noise=renoise,
    )
    sigma = schedule.sigma_for(1000)
    sigma_next = schedule.sigma_for(757)
    pred_video = noise_input - sigma * pred_noise
    expected = (1.0 - sigma_next) * pred_video + sigma_next * renoise
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_dmd_step_requires_noise_for_intermediate_step() -> None:
    schedule = _reference_schedule()
    zeros = np.zeros((1, 16, 3, 8, 8), dtype=np.float32)
    try:
        dmd_step(
            latents=zeros,
            noise_input_latent=zeros,
            pred_noise=zeros,
            schedule=schedule,
            timestep=1000,
            next_timestep=757,
            noise=None,
        )
    except ValueError:
        return
    raise AssertionError("dmd_step must raise when re-noise is requested without noise")
