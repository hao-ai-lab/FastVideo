"""VBench Object Class — GRiT object detection for class matching.

Checks if a target object class is detected in each of 16 sampled frames.
Score = matching_frames / total_frames.
"""

from __future__ import annotations

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("vbench.object_class")
class ObjectClassMetric(BaseMetric):

    name = "vbench.object_class"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["detectron2"]

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def setup(self) -> None:
        if self._model is not None:
            return
        from fastvideo.eval.metrics.vbench._grit_helper import load_grit_model
        # VBench's object_class uses ObjectDet head (init_submodules → "ObjectDet")
        self._model = load_grit_model(self.device, task="ObjectDet")

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        from fastvideo.eval.metrics.vbench._grit_helper import prepare_frames, detect_frames

        video = sample["video"]  # (T, C, H, W)
        aux = sample.get("auxiliary_info") or {}
        if "object" not in aux:
            return self._skip(sample, "missing 'object' in auxiliary_info")

        object_key = aux["object"]
        if " and " in object_key:
            # multiple_objects' territory; skip this row for object_class.
            return self._skip(sample, "'object' contains ' and ' (multi-object)")

        frames_np = prepare_frames(video)
        preds = detect_frames(self._model, frames_np)

        matching = 0
        for frame_pred in preds:
            try:
                obj_set = set(frame_pred[0][2]) if frame_pred else set()
            except (IndexError, TypeError):
                obj_set = set()
            if object_key in obj_set:
                matching += 1

        total = len(preds)
        score = matching / total if total > 0 else 0.0
        return MetricResult(
            name=self.name,
            score=float(score),
            details={
                "matching_frames": matching,
                "total_frames": total
            },
        )
