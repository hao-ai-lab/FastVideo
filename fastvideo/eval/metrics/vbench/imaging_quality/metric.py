"""VBench Imaging Quality — MUSIQ-based per-frame technical quality.

Uses MUSIQ (Multi-Scale Image Quality) from pyiqa.  Frames are resized
so the longer side is at most 512px.  Score = mean(MUSIQ_scores) / 100.
"""

from __future__ import annotations

import torch
from torchvision.transforms.functional import resize

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("vbench.imaging_quality")
class ImagingQualityMetric(BaseMetric):

    name = "vbench.imaging_quality"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["pyiqa"]

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        import pyiqa
        self._model = pyiqa.create_metric("musiq-spaq", device=self.device)
        self._model.eval()

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        video = sample["video"]  # (T, C, H, W)
        T, _, H, W = video.shape

        if max(H, W) > 512:
            scale = 512.0 / max(H, W)
            new_h, new_w = int(H * scale), int(W * scale)
        else:
            new_h, new_w = H, W

        frames = video.to(self.device)
        if (new_h, new_w) != (H, W):
            # antialias=False matches VBench's imaging_quality.transform
            frames = resize(frames, [new_h, new_w], antialias=False)

        chunk = self._chunk_size or 32
        all_scores = []
        for i in range(0, T, chunk):
            scores = self._model(frames[i:i + chunk])
            all_scores.append(scores.squeeze(-1))
        all_scores = torch.cat(all_scores, dim=0)  # (T,)
        return MetricResult(
            name=self.name,
            score=float(all_scores.mean().item()) / 100.0,
            details={"per_frame_raw": all_scores.tolist()},
        )
