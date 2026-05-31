"""VBench Appearance Style — CLIP ViT-B/32 text-image alignment.

Per-frame cosine similarity between CLIP image features and a text
prompt describing the expected style.  Requires ``sample["text_prompt"]``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, center_crop, normalize
from torchvision.transforms import InterpolationMode

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _clip_transform(frames: torch.Tensor) -> torch.Tensor:
    # antialias=False matches VBench's clip_transform (vbench/utils.py:33)
    frames = resize(frames, 224, interpolation=InterpolationMode.BICUBIC, antialias=False)
    frames = center_crop(frames, 224)
    frames = normalize(frames, mean=_CLIP_MEAN, std=_CLIP_STD)
    return frames


@register("vbench.appearance_style")
class AppearanceStyleMetric(BaseMetric):

    name = "vbench.appearance_style"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["clip"]
    backbone = "clip_vit_b32"

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

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
        self._model, _ = clip.load(
            "ViT-B/32",
            device=self.device,
            download_root=str(get_cache_dir() / "clip"),
        )
        self._model.eval()

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        import clip

        video = sample["video"]  # (T, C, H, W)
        text_prompt = sample.get("text_prompt")
        if text_prompt is None:
            return self._skip(sample, "missing text_prompt")

        frames = _clip_transform(video.to(self.device))

        chunk = self._chunk_size or 64
        img_feats = []
        for i in range(0, frames.shape[0], chunk):
            f = self._model.encode_image(frames[i:i + chunk]).float()
            f = F.normalize(f, dim=-1, p=2)
            img_feats.append(f)
        img_feats = torch.cat(img_feats, dim=0)  # (T, D)

        # truncate=True: CLIP context length is 77 tokens; long prompts
        # truncate instead of raising. Matches CLIP's documented convention.
        text_tokens = clip.tokenize([text_prompt], truncate=True).to(self.device)
        text_feat = self._model.encode_text(text_tokens).float()
        text_feat = F.normalize(text_feat, dim=-1, p=2)  # (1, D)

        sims = (img_feats @ text_feat.T).squeeze(-1)  # (T,)
        return MetricResult(
            name=self.name,
            score=float(sims.mean().item()),
            details={"per_frame": sims.tolist()},
        )
