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
        # Per-frame AlexNet feature maps at 1080p run ~500 MB each. A
        # full 121-frame chunk peaks around 60 GB; chunking to 8 frames
        # drops that to ~5 GB with identical numerical output.
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

        # When the worker pre-uploaded inputs (default), these are
        # already on ``self.device`` and the ``.to(...)`` below is a
        # no-op. With ``pre_upload=False`` the worker keeps them on CPU
        # and this metric pays the transfer just like before.
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
