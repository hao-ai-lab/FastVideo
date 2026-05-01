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
    batch_unit = "frame"
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

    def trial_forward(self, batch_size, *, height, width, num_frames):
        h, w = min(height, 512), min(width, 512)
        if max(height, width) > 512:
            scale = 512.0 / max(height, width)
            h, w = int(height * scale), int(width * scale)
        dummy = torch.randn(batch_size, 3, h, w, device=self.device)
        with torch.no_grad():
            self._model(dummy)

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        video = sample["video"]  # (B, T, C, H, W)
        B, T = video.shape[:2]

        # Resize all frames to the same target size
        _, _, H, W = video.shape[1], video.shape[2], video.shape[3], video.shape[4]
        if max(H, W) > 512:
            scale = 512.0 / max(H, W)
            new_h, new_w = int(H * scale), int(W * scale)
        else:
            new_h, new_w = H, W

        frames = video.reshape(B * T, *video.shape[2:]).to(self.device)
        if (new_h, new_w) != (H, W):
            # antialias=False matches VBench's imaging_quality.transform
            frames = resize(frames, [new_h, new_w], antialias=False)

        # Batch through MUSIQ with chunking
        chunk = self._chunk_size or 32
        all_scores = []
        for i in range(0, frames.shape[0], chunk):
            scores = self._model(frames[i:i + chunk])
            all_scores.append(scores.squeeze(-1))
        all_scores = torch.cat(all_scores, dim=0)  # (B*T,)
        all_scores = all_scores.reshape(B, T)

        results = []
        for b in range(B):
            per_frame = all_scores[b].tolist()
            mean_score = float(all_scores[b].mean().item()) / 100.0
            results.append(MetricResult(
                name=self.name,
                score=mean_score,
                details={"per_frame_raw": per_frame},
            ))

        return results
