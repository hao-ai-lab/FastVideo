"""VBench Background Consistency — CLIP ViT-B/32 temporal feature similarity.

Measures background stability via cosine similarity of CLIP features
between consecutive frames and the first frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, center_crop, normalize
from torchvision.transforms import InterpolationMode

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult
from fastvideo.eval.metrics.vbench._utils import consistency_score

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _clip_transform(frames: torch.Tensor) -> torch.Tensor:
    """Apply CLIP preprocessing to (N, C, H, W) float [0,1] tensors."""
    # antialias=False matches VBench's clip_transform (vbench/utils.py:33)
    frames = resize(frames, 224, interpolation=InterpolationMode.BICUBIC, antialias=False)
    frames = center_crop(frames, 224)
    frames = normalize(frames, mean=_CLIP_MEAN, std=_CLIP_STD)
    return frames


@register("vbench.background_consistency")
class BackgroundConsistencyMetric(BaseMetric):

    name = "vbench.background_consistency"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    batch_unit = "frame"
    dependencies = ["clip"]
    backbone = "clip_vit_b32"

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
        import clip
        from fastvideo.eval.models import get_cache_dir
        model, _ = clip.load(
            "ViT-B/32",
            device=self.device,
            download_root=str(get_cache_dir() / "clip"),
        )
        model.eval()
        self._model = model

    def trial_forward(self, batch_size, *, height, width, num_frames):
        dummy = torch.randn(batch_size, 3, 224, 224, device=self.device)
        with torch.no_grad():
            self._model.encode_image(dummy)

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        video = sample["video"]  # (B, T, C, H, W)
        B, T = video.shape[:2]

        frames = video.reshape(B * T, *video.shape[2:]).to(self.device)
        frames = _clip_transform(frames)

        chunk = self._chunk_size or 64
        feats = []
        for i in range(0, frames.shape[0], chunk):
            f = self._model.encode_image(frames[i:i + chunk]).float()
            f = F.normalize(f, dim=-1, p=2)
            feats.append(f)
        all_feats = torch.cat(feats, dim=0).reshape(B, T, -1)

        results = []
        for b in range(B):
            score = consistency_score(all_feats[b])
            results.append(MetricResult(
                name=self.name,
                score=score,
                details={},
            ))

        return results
