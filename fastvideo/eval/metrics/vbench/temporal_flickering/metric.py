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
    batch_unit = "video"

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        video = sample["video"]  # (B, T, C, H, W) float [0, 1]
        B, T = video.shape[:2]

        results = []
        for b in range(B):
            frames = (video[b] * 255.0).to(torch.uint8).cpu().numpy()  # (T, C, H, W)
            # Transpose to (T, H, W, C) for MAE computation
            frames = frames.transpose(0, 2, 3, 1).astype(np.float32)

            if T <= 1:
                results.append(MetricResult(name=self.name, score=1.0, details={}))
                continue

            mae_per_pair = []
            for t in range(T - 1):
                mae = np.mean(np.abs(frames[t] - frames[t + 1]))
                mae_per_pair.append(mae)

            mean_mae = np.mean(mae_per_pair)
            score = (255.0 - mean_mae) / 255.0

            results.append(MetricResult(
                name=self.name,
                score=float(score),
                details={"per_pair_mae": [float(m) for m in mae_per_pair]},
            ))

        return results
