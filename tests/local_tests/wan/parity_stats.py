# SPDX-License-Identifier: Apache-2.0
"""Shared parity diagnostics for Wan vs Diffusers local tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.testing import assert_close

PARITY_ELEMENT_THRESHOLD = 0.02
DIT_PARITY_ATOL = 0.1
DIT_PARITY_RTOL = 0.1
DIT_MAX_ABS_MEAN_DRIFT = 0.05


@dataclass(frozen=True)
class ParityStats:
    label: str
    max_abs: float
    mean_abs: float
    median_abs: float
    p99_abs: float
    num_over_threshold: int
    total_elements: int
    abs_mean_drift_ratio: float

    def as_row(self) -> str:
        over_pct = 100.0 * self.num_over_threshold / max(self.total_elements, 1)
        return (
            f"{self.label}: max={self.max_abs:.6g} mean={self.mean_abs:.6g} "
            f"median={self.median_abs:.6g} p99={self.p99_abs:.6g} "
            f">{PARITY_ELEMENT_THRESHOLD}={self.num_over_threshold}/{self.total_elements} ({over_pct:.1f}%) "
            f"abs_mean_drift={self.abs_mean_drift_ratio:.6g}")


def compute_parity_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    threshold: float = PARITY_ELEMENT_THRESHOLD,
) -> ParityStats:
    diff = (actual.detach().float().cpu() - expected.detach().float().cpu()).abs()
    flat = diff.flatten()
    ref_mean_abs = expected.detach().float().cpu().abs().mean().item()
    drift_ratio = diff.mean().item() / ref_mean_abs if ref_mean_abs > 0 else float("inf")
    p99_index = max(int(0.99 * flat.numel()) - 1, 0)
    return ParityStats(
        label=label,
        max_abs=flat.max().item(),
        mean_abs=flat.mean().item(),
        median_abs=flat.median().item(),
        p99_abs=flat.kthvalue(p99_index + 1).values.item(),
        num_over_threshold=int((flat > threshold).sum().item()),
        total_elements=flat.numel(),
        abs_mean_drift_ratio=drift_ratio,
    )


def assert_dit_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str = "dit",
    *,
    atol: float = DIT_PARITY_ATOL,
    rtol: float = DIT_PARITY_RTOL,
    max_abs_mean_drift: float = DIT_MAX_ABS_MEAN_DRIFT,
) -> ParityStats:
    stats = compute_parity_stats(actual, expected, label, threshold=atol)
    msg = stats.as_row()
    assert_close(actual, expected, atol=atol, rtol=rtol, msg=msg)
    assert stats.abs_mean_drift_ratio < max_abs_mean_drift, (
        f"{msg}; abs_mean_drift {stats.abs_mean_drift_ratio:.6g} >= {max_abs_mean_drift}")
    return stats
