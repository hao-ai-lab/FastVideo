from __future__ import annotations

from typing import Any

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult
from fastvideo.eval.metrics.physics_iq.utils import compute_spatiotemporal_iou, prepare_pair_batch


@register("physics_iq.spatiotemporal_iou")
class SpatiotemporalIoUMetric(BaseMetric):
    name = "physics_iq.spatiotemporal_iou"
    requires_reference = True
    higher_is_better = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._kwargs = kwargs

    def compute(self, sample: dict) -> list[MetricResult]:
        results: list[MetricResult] = []
        for prepared in prepare_pair_batch(sample, prep_kwargs=self._kwargs):
            per_frame = compute_spatiotemporal_iou(prepared.reference_masks, prepared.generated_masks)
            score = sum(per_frame) / len(per_frame)
            results.append(MetricResult(name=self.name, score=score, details={"per_frame": per_frame}))
        return results
