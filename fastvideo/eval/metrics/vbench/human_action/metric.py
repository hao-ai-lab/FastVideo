"""VBench Human Action — UMT ViT-L/16 action classification (Kinetics-400).

Classifies human actions in 16-frame clips. Top-5 predictions with
confidence >= 0.85 are compared against the ground-truth action label.
Score = 1.0 if match found, 0.0 otherwise.
"""

from __future__ import annotations

import os

import torch
from torchvision.transforms.functional import resize, center_crop, normalize

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult
from fastvideo.eval.io.video import extract_frames


# Kinetics-400 class names (loaded lazily)
_CAT_DICT = None


def _load_cat_dict():
    global _CAT_DICT
    if _CAT_DICT is not None:
        return _CAT_DICT
    cat_path = os.path.join(
        os.path.dirname(__file__), "..", "_third_party", "umt",
        "kinetics_400_categories.txt",
    )
    _CAT_DICT = {}
    if os.path.exists(cat_path):
        with open(cat_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    cat, idx = parts
                    _CAT_DICT[idx] = cat.lower()
    return _CAT_DICT


@register("vbench.human_action")
class HumanActionMetric(BaseMetric):

    name = "vbench.human_action"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    batch_unit = "video"
    dependencies = ["timm"]

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
        from timm.models import create_model
        from fastvideo.eval.models import ensure_checkpoint

        ckpt_path = ensure_checkpoint(
            "umt_l16_kinetics400.pth",
            source="OpenGVLab/VBench_Used_Models",
            filename="l16_25m.pth",
        )

        import vbench.third_party.umt.models.modeling_finetune  # noqa: F401

        self._model = create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=400,
            all_frames=16,
            tubelet_size=1,
            use_learnable_pos_emb=False,
            fc_drop_rate=0.0,
            drop_rate=0.0,
            drop_path_rate=0.2,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_checkpoint=False,
            checkpoint_num=16,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(state_dict, strict=False)
        self._model.to(self.device)
        self._model.eval()

    def trial_forward(self, batch_size, *, height, width, num_frames):
        dummy = torch.randn(batch_size, 3, 16, 224, 224, device=self.device)
        with torch.no_grad():
            self._model(dummy)

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        video = sample["video"]  # (B, T, C, H, W) float [0, 1]
        text_prompts = sample.get("text_prompt")
        if text_prompts is None:
            return self._skip(sample, "missing text_prompt with action labels")

        B = video.shape[0]
        cat_dict = _load_cat_dict()
        chunk = self._chunk_size or B

        # Prepare all 16-frame clips: extract, resize, crop, normalize
        all_clips = []
        for b in range(B):
            frames = extract_frames(video[b], 16)  # (16, C, H, W)
            frames = resize(frames, 256, antialias=True)
            frames = center_crop(frames, 224)
            frames = normalize(frames, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            # UMT expects (C, T, H, W)
            all_clips.append(frames.permute(1, 0, 2, 3))
        all_clips = torch.stack(all_clips).to(self.device)  # (B, C, 16, H, W)

        # Batched UMT forward
        all_logits = []
        for i in range(0, B, chunk):
            logits = torch.sigmoid(self._model(all_clips[i:i + chunk]))
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)  # (B, 400)

        results = []
        for b in range(B):
            top_scores, top_indices = torch.topk(all_logits[b:b+1], 5, dim=1)
            top_indices = top_indices.squeeze().tolist()
            top_scores = top_scores.squeeze().tolist()

            predictions = []
            for idx, score in zip(top_indices, top_scores):
                if score >= 0.85:
                    label = cat_dict.get(str(idx), "")
                    predictions.append(label)

            gt_label = text_prompts[b].lower().strip()
            match = any(pred == gt_label for pred in predictions)

            results.append(MetricResult(
                name=self.name,
                score=1.0 if match else 0.0,
                details={"predictions": predictions, "ground_truth": gt_label},
            ))

        return results
