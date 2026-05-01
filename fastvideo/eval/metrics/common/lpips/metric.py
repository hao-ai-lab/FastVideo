from __future__ import annotations

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
    batch_unit = "frame"

    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        self.net = net
        self._model = None

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

    def trial_forward(self, batch_size, *, height, width, num_frames):
        dummy = torch.randn(batch_size, 3, height, width, device=self.device) * 2 - 1
        with torch.no_grad():
            self._model(dummy, dummy)

    def compute(self, sample: dict) -> list[MetricResult]:
        if self._model is None:
            self.setup()

        gen = sample["video"].float().to(self.device)       # (B, T, C, H, W)
        ref = sample["reference"].float().to(self.device)   # (B, T, C, H, W)
        B, T = gen.shape[:2]
        n = min(gen.shape[1], ref.shape[1])
        gen, ref = gen[:, :n], ref[:, :n]

        gen_flat = gen.reshape(B * n, *gen.shape[2:]) * 2.0 - 1.0
        ref_flat = ref.reshape(B * n, *ref.shape[2:]) * 2.0 - 1.0

        # Chunked forward
        chunk = self._chunk_size or len(gen_flat)
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(gen_flat), chunk):
                s = self._model(gen_flat[i:i+chunk], ref_flat[i:i+chunk]).squeeze()
                if s.dim() == 0:
                    s = s.unsqueeze(0)
                all_scores.append(s)
        scores = torch.cat(all_scores).reshape(B, n)

        return [
            MetricResult(
                name=self.name,
                score=float(scores[b].mean()),
                details={"per_frame": scores[b].tolist()},
            )
            for b in range(B)
        ]
