# SPDX-License-Identifier: Apache-2.0
"""Reference-free and forensic metrics for TDM video distillation.

Paired same-noise metrics (latent nearest-neighbour relative MSE / MMD and
paired pixel MS-SSIM) must not be used as quality gates for TDM. In the
standalone experiments the blurry no-guidance student ranked *above* the
coherent distilled student on every paired metric, because those metrics
reward low-variance samples and penalize the paired drift that
distribution-matching objectives produce by design.

Acceptance for TDM therefore uses the reference-free signals in this
module - frame sharpness, latent-cloud diversity, and teacher-cloud
overlap - while the paired/cloud numbers stay forensic-only and are
recorded rather than asserted. The functions are pure torch/numpy so the
tooling and tests can run them anywhere.
"""

from __future__ import annotations

from typing import Any

import torch


def _frames_to_chw(frames: Any) -> torch.Tensor:
    """Normalize frames to ``[N, 3, H, W]`` float; uint8 stays 0-255 scale."""
    tensor = torch.as_tensor(frames)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) frames, got {tuple(tensor.shape)}")
    return tensor.permute(0, 3, 1, 2).float()


def _frames_to_unit_interval(frames: Any) -> torch.Tensor:
    tensor = _frames_to_chw(frames)
    scale = 255.0 if tensor.max() > 1.5 else 1.0
    return tensor / scale


def frame_statistics(frames: Any) -> dict[str, float]:
    """Brightness, contrast, and gradient sharpness of a frame stack.

    ``sharpness`` is the mean squared luminance gradient in the frames'
    own scale (8-bit units for uint8 input). It separates a blurry
    no-guidance ghost from a coherent distilled sample with a much larger
    margin than the plain pixel std that the standalone runs recorded.
    """
    tensor = _frames_to_chw(frames)
    luminance = 0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] + 0.114 * tensor[:, 2]
    gradient_x = luminance[:, :, 1:] - luminance[:, :, :-1]
    gradient_y = luminance[:, 1:, :] - luminance[:, :-1, :]
    return {
        "frames": float(tensor.shape[0]),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "sharpness": float(gradient_x.square().mean() + gradient_y.square().mean()),
    }


def latent_cloud_statistics(cloud: torch.Tensor) -> dict[str, float]:
    """Spread and self-similarity of a sample cloud ``[N, D]``.

    A mode-tightened student shows up as a smaller median pairwise
    distance and a higher mean off-diagonal cosine than the teacher
    cloud; the standalone runs measured a ~0.78 median-distance ratio and
    0.46 versus 0.16 cosine for the single-prompt Wan ladder.
    """
    cloud = cloud.double()
    rows = cloud.shape[0]
    if rows < 2:
        raise ValueError("cloud statistics need at least two samples")
    distances = torch.cdist(cloud, cloud)
    distances.fill_diagonal_(float("inf"))
    upper = torch.triu_indices(rows, rows, offset=1)
    unit = cloud / cloud.norm(dim=1, keepdim=True).clamp_min(1e-6)
    cosine = unit @ unit.t()
    return {
        "rows": float(rows),
        "median_pairwise_distance": float(distances[upper[0], upper[1]].median()),
        "mean_offdiag_cosine": float(cosine[upper[0], upper[1]].mean()),
    }


def teacher_cloud_bandwidth(teacher_cloud: torch.Tensor) -> float:
    """Median squared pairwise distance, the RBF bandwidth convention from
    the standalone cloud reports."""
    distances = torch.pdist(teacher_cloud.double(), p=2)
    return float(distances.square().median())


def cloud_overlap(
    student_cloud: torch.Tensor,
    teacher_cloud: torch.Tensor,
    bandwidth: float | None = None,
) -> dict[str, float]:
    """Cross-cloud RBF overlap and the student/teacher spread ratio."""
    if bandwidth is None:
        bandwidth = teacher_cloud_bandwidth(teacher_cloud)
    student = latent_cloud_statistics(student_cloud)
    teacher = latent_cloud_statistics(teacher_cloud)
    cross = torch.cdist(student_cloud.double(), teacher_cloud.double()).square()
    return {
        "bandwidth": bandwidth,
        "cross_kernel_mean": float(torch.exp(-cross / (2.0 * bandwidth)).mean()),
        "median_distance_ratio": (student["median_pairwise_distance"] / teacher["median_pairwise_distance"]),
        "mean_offdiag_cosine_student": student["mean_offdiag_cosine"],
        "mean_offdiag_cosine_teacher": teacher["mean_offdiag_cosine"],
    }


def nearest_neighbour_relative_mse(student_cloud: torch.Tensor, teacher_cloud: torch.Tensor) -> float:
    """Forensic paired-style metric, kept for comparability only.

    Lower values reward low-variance samples; see the module docstring.
    """
    student = student_cloud.double()
    teacher = teacher_cloud.double()
    cross = torch.cdist(student, teacher).square()
    nearest = cross.min(dim=1)
    teacher_sq = teacher.pow(2).sum(dim=1)
    return float((nearest.values / teacher_sq[nearest.indices]).mean())


def paired_ms_ssim(student_frames: Any, teacher_frames: Any) -> dict[str, float]:
    """Forensic-only paired MS-SSIM; requires the optional ``pytorch_msssim``."""
    try:
        from pytorch_msssim import ms_ssim
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("paired_ms_ssim requires the optional pytorch_msssim package") from error
    left = _frames_to_unit_interval(student_frames)
    right = _frames_to_unit_interval(teacher_frames)
    frames = min(left.shape[0], right.shape[0])
    if frames == 0:
        raise ValueError("paired_ms_ssim needs at least one frame pair")
    return {
        "ms_ssim": float(ms_ssim(left[:frames], right[:frames], data_range=1.0, size_average=True)),
        "frames": float(frames),
    }


__all__ = [
    "cloud_overlap",
    "frame_statistics",
    "latent_cloud_statistics",
    "nearest_neighbour_relative_mse",
    "paired_ms_ssim",
    "teacher_cloud_bandwidth",
]
