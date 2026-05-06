"""VBench Multiple Objects — GRiT detection for dual-object presence.

Checks if BOTH target objects are detected in each of 16 sampled frames.
Score = matching_frames / total_frames.
"""

from __future__ import annotations

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("vbench.multiple_objects")
class MultipleObjectsMetric(BaseMetric):

    name = "vbench.multiple_objects"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    batch_unit = "video"
    dependencies = ["detectron2"]

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def setup(self) -> None:
        if self._model is not None:
            return
        from fastvideo.eval.metrics.vbench._grit_helper import load_grit_model
        # VBench's multiple_objects uses ObjectDet head
        self._model = load_grit_model(self.device, task="ObjectDet")

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        from fastvideo.eval.metrics.vbench._grit_helper import prepare_frames, detect_frames

        video = sample["video"]  # (B, T, C, H, W)
        aux = sample.get("auxiliary_info")

        B = video.shape[0]
        results = []

        for b in range(B):
            aux_b = aux[b] if isinstance(aux, list) else aux
            if not aux_b or "object" not in aux_b:
                results.append(MetricResult(
                    name=self.name, score=None,
                    details={"skipped": "missing 'object' in auxiliary_info"}))
                continue
            object_info = aux_b["object"]
            if " and " not in object_info:
                # multiple_objects expects "<a> and <b>"; single objects are
                # the object_class metric's territory — skip this row.
                results.append(MetricResult(
                    name=self.name, score=None,
                    details={"skipped": "'object' lacks ' and ' separator"}))
                continue
            key_a, key_b = [k.strip() for k in object_info.split(" and ")]

            frames_np = prepare_frames(video[b])
            preds = detect_frames(self._model, frames_np)

            matching = 0
            for frame_pred in preds:
                try:
                    obj_set = set(frame_pred[0][2]) if frame_pred else set()
                except (IndexError, TypeError):
                    obj_set = set()
                if key_a in obj_set and key_b in obj_set:
                    matching += 1

            total = len(preds)
            score = matching / total if total > 0 else 0.0
            results.append(MetricResult(
                name=self.name,
                score=float(score),
                details={"matching_frames": matching, "total_frames": total},
            ))

        return results
