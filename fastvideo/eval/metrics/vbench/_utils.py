"""Shared utilities for VBench metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def consistency_score(features: torch.Tensor) -> float:
    """VBench-style temporal consistency from (T, D) L2-normalized features.

    For each frame t > 0, computes:
        sim = (cos(f[t], f[t-1]) + cos(f[t], f[0])) / 2, clamped >= 0

    Returns the mean similarity across all t > 0.
    """
    if features.shape[0] <= 1:
        return 1.0

    first = features[0:1]  # (1, D)
    total_sim = 0.0
    count = 0
    for t in range(1, features.shape[0]):
        curr = features[t:t + 1]
        prev = features[t - 1:t]
        sim_prev = max(0.0, F.cosine_similarity(prev, curr).item())
        sim_first = max(0.0, F.cosine_similarity(first, curr).item())
        total_sim += (sim_prev + sim_first) / 2
        count += 1

    return total_sim / count
