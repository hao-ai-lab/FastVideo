"""VBench Aesthetic Quality — CLIP ViT-L/14 + LAION aesthetic predictor.

Encodes frames through CLIP, passes L2-normalized features through a
linear aesthetic head (768 → 1), and averages scores / 10.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, center_crop, normalize
from torchvision.transforms import InterpolationMode

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_AESTHETIC_URL = "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth"


def _clip_transform(frames: torch.Tensor) -> torch.Tensor:
    # antialias=False matches VBench's clip_transform (vbench/utils.py:33)
    frames = resize(frames, 224, interpolation=InterpolationMode.BICUBIC, antialias=False)
    frames = center_crop(frames, 224)
    frames = normalize(frames, mean=_CLIP_MEAN, std=_CLIP_STD)
    return frames


@register("vbench.aesthetic_quality")
class AestheticQualityMetric(BaseMetric):

    name = "vbench.aesthetic_quality"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["clip"]
    backbone = "clip_vit_l14"

    def __init__(self) -> None:
        super().__init__()
        self._clip_model: Any = None
        self._aesthetic_head: Any = None

    def to(self, device):
        super().to(device)
        if self._clip_model is not None:
            self._clip_model = self._clip_model.to(self.device)
        if self._aesthetic_head is not None:
            self._aesthetic_head = self._aesthetic_head.to(self.device)
        return self

    def setup(self) -> None:
        if self._clip_model is not None:
            return

        import clip
        from fastvideo.eval.models import ensure_checkpoint, get_cache_dir
        self._clip_model, _ = clip.load(
            "ViT-L/14",
            device=self.device,
            download_root=str(get_cache_dir() / "clip"),
        )
        self._clip_model.eval()

        # Load LAION aesthetic head
        ckpt_path = ensure_checkpoint(
            "sa_0_4_vit_l_14_linear.pth",
            source=_AESTHETIC_URL,
        )
        self._aesthetic_head = nn.Linear(768, 1)
        self._aesthetic_head.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        self._aesthetic_head.to(self.device)
        self._aesthetic_head.eval()

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        video = sample["video"]  # (T, C, H, W)
        frames = _clip_transform(video.to(self.device))

        chunk = self._chunk_size or 32
        scores_list = []
        for i in range(0, frames.shape[0], chunk):
            feats = self._clip_model.encode_image(frames[i:i + chunk]).float()
            feats = F.normalize(feats, dim=-1, p=2)
            scores_list.append(self._aesthetic_head(feats).squeeze(-1))

        all_scores = torch.cat(scores_list, dim=0) / 10.0  # (T,)
        return MetricResult(
            name=self.name,
            score=float(all_scores.mean().item()),
            details={"per_frame": all_scores.tolist()},
        )
