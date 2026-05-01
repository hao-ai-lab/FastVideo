"""VBench Subject Consistency — DINO ViT-B/16 temporal feature similarity.

Measures how well the main subject maintains its appearance throughout
the video via cosine similarity of DINO features between consecutive
frames and the first frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, normalize

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult
from fastvideo.eval.metrics.vbench._utils import consistency_score

# ImageNet normalization (used by DINO)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


@register("vbench.subject_consistency")
class SubjectConsistencyMetric(BaseMetric):

    name = "vbench.subject_consistency"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    batch_unit = "frame"
    backbone = "dino_vitb16"

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
        model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        model.to(self.device)
        model.eval()
        self._model = model

    def trial_forward(self, batch_size, *, height, width, num_frames):
        dummy = torch.randn(batch_size, 3, 224, 224, device=self.device)
        with torch.no_grad():
            self._model(dummy)

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        video = sample["video"]  # (B, T, C, H, W)
        B, T = video.shape[:2]

        # Flatten, transform, extract features
        frames = video.reshape(B * T, *video.shape[2:]).to(self.device)
        # antialias=False matches VBench's dino_transform (vbench/utils.py:50)
        frames = resize(frames, 224, antialias=False)
        frames = normalize(frames, mean=_MEAN, std=_STD)

        chunk = self._chunk_size or 64
        feats = []
        for i in range(0, frames.shape[0], chunk):
            f = self._model(frames[i:i + chunk])
            f = F.normalize(f, dim=-1, p=2)
            feats.append(f)
        all_feats = torch.cat(feats, dim=0)  # (B*T, D)
        all_feats = all_feats.reshape(B, T, -1)  # (B, T, D)

        results = []
        for b in range(B):
            score = consistency_score(all_feats[b])
            results.append(MetricResult(
                name=self.name,
                score=score,
                details={},
            ))

        return results
