"""VBench Temporal Flickering — measures frame-to-frame stability.

Score = (255 - mean_MAE) / 255, where MAE is computed between consecutive
frames in uint8 [0, 255] space.  Higher = less flickering.
"""

from __future__ import annotations

import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("vbench.temporal_flickering")
class TemporalFlickeringMetric(BaseMetric):

    name = "vbench.temporal_flickering"
    requires_reference = False
    higher_is_better = True
    needs_gpu = False

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        video = sample["video"]  # (T, C, H, W) [0, 1]
        T = video.shape[0]
        if T <= 1:
            return MetricResult(name=self.name, score=1.0, details={})

        frames = (video * 255.0).to(torch.uint8).cpu().numpy()
        frames = frames.transpose(0, 2, 3, 1).astype(np.float32)
        mae_per_pair = [float(np.mean(np.abs(frames[t] - frames[t + 1]))) for t in range(T - 1)]
        mean_mae = float(np.mean(mae_per_pair))
        return MetricResult(
            name=self.name,
            score=(255.0 - mean_mae) / 255.0,
            details={"per_pair_mae": mae_per_pair},
        )
