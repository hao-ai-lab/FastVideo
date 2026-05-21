"""VBench Spatial Relationship — GRiT detection + bbox position scoring.

Detects two target objects via GRiT and checks if their bounding boxes
satisfy the expected spatial relationship (left/right/above/below).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


def _get_position_score(locality: str, obj1: list, obj2: list, iou_threshold: float = 0.1) -> float:
    """Score spatial relationship between two bboxes [x0, y0, x1, y1].

    Matching VBench's get_position_score() exactly.
    """
    box1_center = ((obj1[0] + obj1[2]) / 2, (obj1[1] + obj1[3]) / 2)
    box2_center = ((obj2[0] + obj2[2]) / 2, (obj2[1] + obj2[3]) / 2)

    x_distance = box2_center[0] - box1_center[0]
    y_distance = box2_center[1] - box1_center[1]

    # IoU
    x_overlap = max(0, min(obj1[2], obj2[2]) - max(obj1[0], obj2[0]))
    y_overlap = max(0, min(obj1[3], obj2[3]) - max(obj1[1], obj2[1]))
    intersection = x_overlap * y_overlap
    area1 = (obj1[2] - obj1[0]) * (obj1[3] - obj1[1])
    area2 = (obj2[2] - obj2[0]) * (obj2[3] - obj2[1])
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0

    if "right" in locality or "left" in locality:
        if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
            return 1.0
        elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
            return iou_threshold / iou
        return 0.0
    elif "bottom" in locality or "top" in locality:
        if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
            return 1.0
        elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
            return iou_threshold / iou
        return 0.0
    return 0.0


@register("vbench.spatial_relationship")
class SpatialRelationshipMetric(BaseMetric):

    name = "vbench.spatial_relationship"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["detectron2"]

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    def setup(self) -> None:
        if self._model is not None:
            return
        from fastvideo.eval.metrics.vbench._grit_helper import load_grit_model
        # VBench's spatial_relationship uses ObjectDet head and matches
        # pred[0] against class names like "person"/"grass"
        self._model = load_grit_model(self.device, task="ObjectDet")

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        from fastvideo.eval.metrics.vbench._grit_helper import prepare_frames

        video = sample["video"]  # (T, C, H, W)
        aux = sample.get("auxiliary_info") or {}
        if "spatial_relationship" not in aux:
            return self._skip(sample, "missing 'spatial_relationship' in auxiliary_info")

        sp_info = aux["spatial_relationship"]
        try:
            key_a = sp_info["object_a"]
            key_b = sp_info["object_b"]
            relation = sp_info["relationship"]
        except (KeyError, TypeError):
            return self._skip(sample, "spatial_relationship missing object_a/object_b/relationship")

        frames_np = prepare_frames(video)

        preds = []
        for frame in frames_np:
            ret = self._model.run_caption_tensor(frame)
            frame_dets = []
            if len(ret[0]) > 0:
                for info in ret[0]:
                    frame_dets.append([info[0], info[1]])  # (caption, bbox)
            preds.append(frame_dets)

        # Score each frame (matching VBench's check_generate).
        frame_scores: list[float] = []
        for frame_pred in preds:
            obj_bboxes = [item[1] for item in frame_pred if item[0] == key_a or item[0] == key_b]

            cur_scores = [0.0]
            for i in range(len(obj_bboxes) - 1):
                for j in range(i + 1, len(obj_bboxes)):
                    cur_scores.append(_get_position_score(
                        relation,
                        obj_bboxes[i],
                        obj_bboxes[j],
                    ))
            frame_scores.append(max(cur_scores))

        score = float(np.mean(frame_scores)) if frame_scores else 0.0
        return MetricResult(
            name=self.name,
            score=score,
            details={"per_frame": frame_scores},
        )
