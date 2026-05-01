from __future__ import annotations

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("common.psnr")
class PSNRMetric(BaseMetric):
    name = "common.psnr"
    requires_reference = True
    higher_is_better = True
    needs_gpu = False

    def __init__(self, max_val: float = 1.0) -> None:
        super().__init__()
        self.max_val = max_val

    def compute(self, sample: dict) -> list[MetricResult]:
        gen = sample["video"].float()       # (B, T, C, H, W)
        ref = sample["reference"].float()   # (B, T, C, H, W)
        B, T = gen.shape[:2]
        n = min(gen.shape[1], ref.shape[1])
        gen, ref = gen[:, :n], ref[:, :n]

        # Vectorized: per-frame MSE across entire batch at once
        mse = ((gen - ref) ** 2).mean(dim=(2, 3, 4))  # (B, T)
        psnr = 10.0 * torch.log10(self.max_val**2 / mse.clamp(min=1e-10))  # (B, T)

        return [
            MetricResult(
                name=self.name,
                score=psnr[b].mean().item(),
                details={"per_frame": psnr[b].tolist()},
            )
            for b in range(B)
        ]
