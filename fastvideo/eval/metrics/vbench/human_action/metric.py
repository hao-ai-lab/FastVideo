"""VBench Human Action — UMT ViT-L/16 action classification (Kinetics-400).

Classifies human actions in 16-frame clips. Top-5 predictions with
confidence >= 0.85 are compared against the ground-truth action label.
Score = 1.0 if match found, 0.0 otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.transforms.functional import resize, center_crop, normalize

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult
from fastvideo.eval.io.video import extract_frames

# Kinetics-400 class names (loaded lazily). The label file ships inside
# the upstream vbench submodule.
_CAT_DICT: dict[str, str] | None = None


def _load_cat_dict() -> dict[str, str]:
    global _CAT_DICT
    if _CAT_DICT is not None:
        return _CAT_DICT
    import vbench.third_party.umt as _umt_pkg
    cat_path = (Path(_umt_pkg.__file__).resolve().parent / "kinetics_400_categories.txt")
    out: dict[str, str] = {}
    with cat_path.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                cat, idx = parts
                out[idx] = cat.lower()
    _CAT_DICT = out
    return _CAT_DICT


@register("vbench.human_action")
class HumanActionMetric(BaseMetric):

    name = "vbench.human_action"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["timm"]

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
        from timm.models import create_model
        from fastvideo.eval.models import ensure_checkpoint

        ckpt_path = ensure_checkpoint(
            "umt_l16_kinetics400.pth",
            source="OpenGVLab/VBench_Used_Models",
            filename="l16_ptk710_ftk710_ftk400_f16_res224.pth",
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

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        video = sample["video"]  # (T, C, H, W) [0, 1]
        text_prompt = sample.get("text_prompt")
        if text_prompt is None:
            return self._skip(sample, "missing text_prompt with action labels")

        cat_dict = _load_cat_dict()

        frames = extract_frames(video, 16)  # (16, C, H, W)
        frames = resize(frames, 256, antialias=True)
        frames = center_crop(frames, 224)
        frames = normalize(frames, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # UMT expects (C, T, H, W); add a leading batch dim of 1.
        clip_in = frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        logits = torch.sigmoid(self._model(clip_in))  # (1, 400)
        top_scores, top_indices = torch.topk(logits[0], 5)
        top_indices = top_indices.tolist()
        top_scores = top_scores.tolist()

        predictions = [
            cat_dict.get(str(idx), "") for idx, score in zip(top_indices, top_scores, strict=False) if score >= 0.85
        ]
        gt_label = text_prompt.lower().strip()
        match = any(pred == gt_label for pred in predictions)
        return MetricResult(
            name=self.name,
            score=1.0 if match else 0.0,
            details={
                "predictions": predictions,
                "ground_truth": gt_label
            },
        )
