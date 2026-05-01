from __future__ import annotations

from typing import Any

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult
from fastvideo.eval.metrics.physics_iq.utils import compute_weighted_spatial_iou, prepare_pair_batch


@register("physics_iq.weighted_spatial_iou")
class WeightedSpatialIoUMetric(BaseMetric):
    name = "physics_iq.weighted_spatial_iou"
    requires_reference = True
    higher_is_better = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._kwargs = kwargs

    def compute(self, sample: dict) -> list[MetricResult]:
        results: list[MetricResult] = []
        for prepared in prepare_pair_batch(sample, prep_kwargs=self._kwargs):
            score = compute_weighted_spatial_iou(prepared.reference_masks, prepared.generated_masks)
            results.append(MetricResult(name=self.name, score=score, details={}))
        return results
