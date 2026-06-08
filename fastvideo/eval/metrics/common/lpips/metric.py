from __future__ import annotations

from typing import Any

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("common.lpips")
class LPIPSMetric(BaseMetric):
    name = "common.lpips"
    requires_reference = True
    higher_is_better = False
    needs_gpu = True
    dependencies = ["lpips"]

    def __init__(self, net: str = "alex", chunk_size: int = 8) -> None:
        super().__init__()
        self.net = net
        # Chunk the per-frame forward: AlexNet feature maps at 1080p
        # peak ~60 GB on a 121-frame clip; chunk=8 caps it at ~5 GB.
        self._chunk_size = chunk_size
        self._model: Any = None

    def to(self, device: str | torch.device) -> LPIPSMetric:
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        import lpips as lpips_lib
        self._model = lpips_lib.LPIPS(net=self.net).to(self.device)
        self._model.eval()

    def compute(self, sample: dict) -> MetricResult:
        if self._model is None:
            self.setup()

        gen = sample["video"].float().to(self.device, non_blocking=True)
        ref = sample["reference"].float().to(self.device, non_blocking=True)

        n = min(gen.shape[0], ref.shape[0])
        gen, ref = gen[:n] * 2.0 - 1.0, ref[:n] * 2.0 - 1.0

        chunk = self._chunk_size or n
        all_scores = []
        with torch.no_grad():
            for i in range(0, n, chunk):
                s = self._model(gen[i:i + chunk], ref[i:i + chunk]).squeeze()
                if s.dim() == 0:
                    s = s.unsqueeze(0)
                all_scores.append(s)
        scores = torch.cat(all_scores)  # (n,)

        return MetricResult(
            name=self.name,
            score=float(scores.mean()),
            details={"per_frame": scores.tolist()},
        )
