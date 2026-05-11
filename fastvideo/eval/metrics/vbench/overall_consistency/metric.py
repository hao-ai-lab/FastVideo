"""VBench Overall Consistency — ViCLIP text-video alignment.

Encodes 8 sampled video frames via ViCLIP vision encoder and a text
prompt via ViCLIP text encoder, then computes cosine similarity.
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
from fastvideo.eval.io.video import extract_frames

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _clip_transform(frames: torch.Tensor) -> torch.Tensor:
    frames = resize(frames, 224, interpolation=InterpolationMode.BICUBIC, antialias=True)
    frames = center_crop(frames, 224)
    frames = normalize(frames, mean=_CLIP_MEAN, std=_CLIP_STD)
    return frames


@register("vbench.overall_consistency")
class OverallConsistencyMetric(BaseMetric):

    name = "vbench.overall_consistency"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["timm", "einops", "clip"]
    backbone = "viclip"

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._tokenizer: Any = None

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        from vbench.third_party.ViCLIP.viclip import ViCLIP
        from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer

        # ViCLIP's tokenizer reuses OpenAI CLIP's BPE vocab. The file is
        # bundled with the ``openai-clip`` pip package (an ``[eval]`` extra)
        # — no separate download is needed; the model loader only handles
        # the actual .pth weights.
        from clip.simple_tokenizer import default_bpe
        self._tokenizer = SimpleTokenizer(default_bpe())

        from fastvideo.eval.models import ensure_checkpoint
        ckpt = ensure_checkpoint(
            "ViClip-InternVid-10M-FLT.pth",
            source="OpenGVLab/VBench_Used_Models",
            filename="ViClip-InternVid-10M-FLT.pth",
        )

        self._model = ViCLIP(tokenizer=self._tokenizer, pretrain=ckpt)
        self._model.to(self.device)
        self._model.eval()

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        video = sample["video"]  # (T, C, H, W)
        text_prompt = sample.get("text_prompt")
        if text_prompt is None:
            return self._skip(sample, "missing text_prompt")

        frames = _clip_transform(extract_frames(video, 8))  # (8, C, H, W)
        clip_in = frames.unsqueeze(0).to(self.device)  # (1, 8, C, H, W)

        vid_feat = self._model.encode_vision(clip_in, test=True).float()
        vid_feat = F.normalize(vid_feat, dim=-1, p=2)  # (1, D)

        text_feat = self._model.encode_text(text_prompt).float()
        text_feat = F.normalize(text_feat, dim=-1, p=2)
        score = float((vid_feat @ text_feat.T)[0][0].cpu())
        return MetricResult(name=self.name, score=score, details={})
