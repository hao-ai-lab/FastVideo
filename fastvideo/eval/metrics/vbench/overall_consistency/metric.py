"""VBench Overall Consistency — ViCLIP text-video alignment.

Encodes 8 sampled video frames via ViCLIP vision encoder and a text
prompt via ViCLIP text encoder, then computes cosine similarity.
"""

from __future__ import annotations

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
    batch_unit = "video"
    dependencies = ["timm", "einops", "clip"]
    backbone = "viclip"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._tokenizer = None

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

    def trial_forward(self, batch_size, *, height, width, num_frames):
        dummy = torch.randn(batch_size, 8, 3, 224, 224, device=self.device)
        with torch.no_grad():
            self._model.encode_vision(dummy, test=True)

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        video = sample["video"]  # (B, T, C, H, W)
        text_prompts = sample.get("text_prompt")
        if text_prompts is None:
            return self._skip(sample, "missing text_prompt")

        B = video.shape[0]
        chunk = self._chunk_size or B

        # Prepare all 8-frame clips and transform
        all_clips = []
        for b in range(B):
            frames = extract_frames(video[b], 8)  # (8, C, H, W)
            frames = _clip_transform(frames)
            all_clips.append(frames)
        all_clips = torch.stack(all_clips).to(self.device)  # (B, 8, C, H, W)

        # Batched ViCLIP vision encoding
        all_vid_feats = []
        for i in range(0, B, chunk):
            vid_feat = self._model.encode_vision(all_clips[i:i + chunk], test=True).float()
            vid_feat = F.normalize(vid_feat, dim=-1, p=2)
            all_vid_feats.append(vid_feat)
        all_vid_feats = torch.cat(all_vid_feats, dim=0)  # (B, D)

        results = []
        for b in range(B):
            text_feat = self._model.encode_text(text_prompts[b]).float()
            text_feat = F.normalize(text_feat, dim=-1, p=2)
            score = float((all_vid_feats[b:b+1] @ text_feat.T)[0][0].cpu())
            results.append(MetricResult(name=self.name, score=score, details={}))

        return results
