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
    # PSNR is `((gen - ref)**2).mean(...)` plus log — memory-bandwidth-
    # bound on host (~6 GB read + 3 GB write per video pair at 1080p ×
    # 121 fr). Trivial on GPU and frees the host bus for the loader.
    needs_gpu = True

    def __init__(self, max_val: float = 1.0, chunk_size: int = 32) -> None:
        super().__init__()
        self.max_val = max_val
        # (gen - ref)**2 at 1080p × 121 fr allocates a full ~3 GB
        # intermediate. chunk=32 caps that at ~800 MB with identical
        # numerical output.
        self._chunk_size = chunk_size

    def compute(self, sample: dict) -> MetricResult:
        gen = sample["video"].float().to(self.device)  # (T, C, H, W)
        ref = sample["reference"].float().to(self.device)
        n = min(gen.shape[0], ref.shape[0])
        gen, ref = gen[:n], ref[:n]

        # Per-frame MSE → PSNR, chunked so the squared-diff intermediate
        # never holds the whole clip at once.
        chunk = self._chunk_size or n
        mse_parts = []
        for i in range(0, n, chunk):
            g = gen[i:i + chunk]
            r = ref[i:i + chunk]
            mse_parts.append(((g - r)**2).mean(dim=(1, 2, 3)))
        mse = torch.cat(mse_parts)  # (T,)
        psnr = 10.0 * torch.log10(self.max_val**2 / mse.clamp(min=1e-10))

        return MetricResult(
            name=self.name,
            score=psnr.mean().item(),
            details={"per_frame": psnr.tolist()},
        )
