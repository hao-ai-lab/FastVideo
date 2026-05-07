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

    def compute(self, sample: dict) -> MetricResult:
        gen = sample["video"].float()  # (T, C, H, W)
        ref = sample["reference"].float()
        n = min(gen.shape[0], ref.shape[0])
        gen, ref = gen[:n], ref[:n]

        # Per-frame MSE → PSNR.
        mse = ((gen - ref)**2).mean(dim=(1, 2, 3))  # (T,)
        psnr = 10.0 * torch.log10(self.max_val**2 / mse.clamp(min=1e-10))

        return MetricResult(
            name=self.name,
            score=psnr.mean().item(),
            details={"per_frame": psnr.tolist()},
        )
