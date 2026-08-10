# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Wan2.2 warped DMD schedule construction."""

from __future__ import annotations

from fastvideo.mlx_runtime.wan22_sample import build_wan22_dmd_schedule


def test_warped_dmd_timesteps_differ_from_raw_indices() -> None:
    schedule, warped = build_wan22_dmd_schedule([1000, 757, 522], flow_shift=5.0, warp_denoising_step=True)
    _, raw = build_wan22_dmd_schedule([1000, 757, 522], flow_shift=5.0, warp_denoising_step=False)
    assert raw == [1000.0, 757.0, 522.0]
    # Warping maps step indices into the continuous flow-match schedule.
    assert warped[0] == 1000.0
    assert warped[1] != 757.0
    assert warped[2] != 522.0
    # Sigmas are monotone-ish decreasing along the schedule.
    sigmas = [schedule.sigma_for(t) for t in warped]
    assert sigmas[0] >= sigmas[1] >= sigmas[2]
