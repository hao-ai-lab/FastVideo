"""VBench Temporal Style — ViCLIP text-video alignment (style focus).

Identical logic to overall_consistency — same ViCLIP cosine similarity.
The difference is semantic: overall_consistency measures general prompt
alignment while temporal_style measures style consistency over time.
VBench uses different prompts for each from its metadata JSON.
"""

from __future__ import annotations

from fastvideo.eval.registry import register
from fastvideo.eval.metrics.vbench.overall_consistency.metric import OverallConsistencyMetric


@register("vbench.temporal_style")
class TemporalStyleMetric(OverallConsistencyMetric):
    name = "vbench.temporal_style"
