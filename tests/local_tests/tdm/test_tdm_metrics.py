"""Tests for the reference-free TDM metrics.

The paired-metric inversion (blurry low-variance samples ranking above
coherent ones) is pinned here as documented behaviour so it cannot be
reintroduced as an acceptance gate by accident.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastvideo.train.utils.tdm_metrics import (
    cloud_overlap,
    frame_statistics,
    latent_cloud_statistics,
    nearest_neighbour_relative_mse,
    teacher_cloud_bandwidth,
)


def _sharp_frames(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rows = torch.arange(32).float().reshape(1, -1, 1, 1)
    columns = torch.arange(32).float().reshape(1, 1, -1, 1)
    checker = ((rows + columns) % 2) * 255.0
    noise = torch.randn(4, 32, 32, 3, generator=generator) * 5.0
    frames = (checker + noise).clamp(0.0, 255.0)
    return frames.to(torch.uint8)


def test_frame_statistics_separates_blur_from_sharpness() -> None:
    sharp = _sharp_frames()
    blurred = F.avg_pool2d(sharp.permute(0, 3, 1, 2).float(), kernel_size=5, stride=1,
                           padding=2).permute(0, 2, 3, 1).to(torch.uint8)
    flat = torch.full_like(sharp, 128)

    sharp_stats = frame_statistics(sharp)
    blurred_stats = frame_statistics(blurred)
    flat_stats = frame_statistics(flat)

    assert sharp_stats["sharpness"] > blurred_stats["sharpness"] > flat_stats["sharpness"]
    assert flat_stats["sharpness"] == 0.0
    assert flat_stats["std"] == 0.0
    assert sharp_stats["std"] > blurred_stats["std"] > 0.0


def test_latent_cloud_statistics_detects_mode_tightening() -> None:
    generator = torch.Generator().manual_seed(3)
    spread = torch.randn(16, 32, generator=generator)
    collapsed = spread[:1].repeat(16, 1) + 0.01 * torch.randn(16, 32, generator=generator)

    spread_stats = latent_cloud_statistics(spread)
    collapsed_stats = latent_cloud_statistics(collapsed)

    assert collapsed_stats["median_pairwise_distance"] < spread_stats["median_pairwise_distance"]
    assert collapsed_stats["mean_offdiag_cosine"] > spread_stats["mean_offdiag_cosine"]


def test_cloud_overlap_tracks_shift_and_spread() -> None:
    generator = torch.Generator().manual_seed(4)
    teacher = torch.randn(16, 32, generator=generator)
    shifted = teacher + 20.0
    tightened = 0.3 * teacher + 0.7 * teacher.mean(dim=0, keepdim=True)

    bandwidth = teacher_cloud_bandwidth(teacher)
    assert bandwidth > 0.0

    identical = cloud_overlap(teacher, teacher, bandwidth)
    shifted_stats = cloud_overlap(shifted, teacher, bandwidth)
    tightened_stats = cloud_overlap(tightened, teacher, bandwidth)

    assert identical["cross_kernel_mean"] > 0.5
    assert shifted_stats["cross_kernel_mean"] < 1e-6
    assert tightened_stats["median_distance_ratio"] < 1.0
    assert tightened_stats["mean_offdiag_cosine_student"] > tightened_stats["mean_offdiag_cosine_teacher"]


def test_paired_nearest_neighbour_metric_rewards_low_variance() -> None:
    """The documented inversion: halving the spread beats matching the mean."""
    generator = torch.Generator().manual_seed(5)
    teacher = torch.randn(32, 64, generator=generator)
    tightened = teacher * 0.5
    shifted = teacher + torch.randn(32, 64, generator=generator)

    tightened_score = nearest_neighbour_relative_mse(tightened, teacher)
    shifted_score = nearest_neighbour_relative_mse(shifted, teacher)

    assert tightened_score < 0.3
    assert shifted_score > tightened_score
    # The tightened cloud is also less diverse, which is why the paired
    # score cannot stand in for quality.
    assert latent_cloud_statistics(tightened)["median_pairwise_distance"] < 0.6 * (
        latent_cloud_statistics(teacher)["median_pairwise_distance"])
