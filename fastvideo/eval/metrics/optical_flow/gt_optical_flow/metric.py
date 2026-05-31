"""Compare optical flow extracted from a generated video against optical
flow extracted from a ground-truth reference video.

Both flows are produced by the same ``ptlflow`` model (default
``dpflow``/``things``). The resulting per-pixel / per-frame / temporal
metric set is identical to ``synthetic_optical_flow`` — only the way the
*reference* flow is constructed differs.
"""

from __future__ import annotations

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.metrics.optical_flow._shared import (
    aggregate_temporal,
    compute_frame_metrics,
    extract_video_flows,
    load_ptlflow_model,
)
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("optical_flow.gt_optical_flow")
class GtOpticalFlowMetric(BaseMetric):
    """Per-pixel / per-frame / temporal flow comparison vs. a reference video.

    The headline ``score`` is ``pixel_epe_mean_mean`` (lower is better);
    every other scalar lives in ``details`` so downstream consumers can
    pick whichever one they care about.
    """

    name = "optical_flow.gt_optical_flow"
    requires_reference = True
    higher_is_better = False
    needs_gpu = True
    backbone = "optical_flow"
    dependencies = ["ptlflow"]

    def __init__(
        self,
        model_name: str = "dpflow",
        ckpt: str = "things",
        min_mag: float = 0.5,
        max_mag_pct: float = 80.0,
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.ckpt = ckpt
        self.min_mag = min_mag
        self.max_mag_pct = max_mag_pct
        self.grid_size = grid_size
        self._model = None
        # One frame-pair per DPFlow forward: the cost volume is ~4 GB
        # per pair at 1080p, so batching OOMs. Bump for low-res inputs.
        self._chunk_size = 1

    def to(self, device: str | torch.device) -> GtOpticalFlowMetric:
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        self._model = load_ptlflow_model(self.model_name, self.ckpt, self.device)

    def compute(self, sample: dict) -> MetricResult:
        if self._model is None:
            self.setup()

        gen_video = sample["video"].float()  # (T, C, H, W)
        ref_video = sample["reference"].float()
        n = min(gen_video.shape[0], ref_video.shape[0])
        gen_video, ref_video = gen_video[:n], ref_video[:n]
        if n < 2:
            raise ValueError("Need at least 2 frames to compute optical flow")

        chunk = self._chunk_size or 16
        gen_flows = extract_video_flows(
            self._model,
            gen_video,
            chunk=chunk,
            device=self.device,
        )
        ref_flows = extract_video_flows(
            self._model,
            ref_video,
            chunk=chunk,
            device=self.device,
        )
        per_frame = [
            compute_frame_metrics(
                rf,
                gf,
                grid_size=self.grid_size,
                min_mag=self.min_mag,
                max_mag_pct=self.max_mag_pct,
            ) for rf, gf in zip(ref_flows, gen_flows, strict=False)
        ]
        summary = aggregate_temporal(per_frame)
        score = summary.get("pixel_epe_mean_mean")
        details = dict(summary)
        details["per_frame_metrics"] = per_frame
        return MetricResult(
            name=self.name,
            score=float(score) if score is not None else None,
            details=details,
        )
