"""Compare optical flow extracted from a generated video against optical
flow synthesized analytically from per-frame actions.

The reference flow is *not* observed from a ground-truth video — it's
predicted from the action stream via a third-person camera-kinematics
model (Longuet-Higgins linearization + off-pivot translation correction;
no depth). Observed flow comes from the same ``ptlflow`` model used by
``gt_optical_flow``, and the two are compared with the identical metric
set, so scores are directly comparable across the two metrics.

Required sample keys
--------------------
``video``
    ``(B, T, C, H, W)`` float in ``[0, 1]``.
``actions``
    ``dict`` (or list-of-dicts of length B) with two ``np.ndarray`` keys:

    * ``keyboard`` of shape ``(T, 6)`` — ``[W, S, A, D, turn_left, turn_right]``
    * ``mouse`` of shape ``(T, 2)`` — ``[pitch, yaw]``
``calibration``
    Either a path to a ``ThirdPersonCalibration`` JSON file, or a dict
    of fitted parameters. May also be set once at construction time via
    ``calibration_path=`` and reused across samples.

Optional sample keys
--------------------
``mouse_pitch_sign``
    ``+1`` (default) or ``-1`` if the dataset's mouse-pitch sign is
    flipped (mhuo's data carries this in metadata).
"""

from __future__ import annotations

from pathlib import Path

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.metrics.optical_flow._shared import (
    aggregate_temporal,
    compute_frame_metrics,
    extract_video_flows,
    load_ptlflow_model,
)
from fastvideo.eval.metrics.optical_flow.synthetic_optical_flow._thirdperson import (
    ThirdPersonCalibration,
    ThirdPersonFlowGenerator,
    load_calibration,
)
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


def _resolve_calibration(obj: str | Path | dict | ThirdPersonCalibration, ) -> ThirdPersonCalibration:
    if isinstance(obj, ThirdPersonCalibration):
        return obj
    if isinstance(obj, dict):
        return ThirdPersonCalibration.from_dict(obj)
    return load_calibration(obj)


@register("optical_flow.synthetic_optical_flow")
class SyntheticOpticalFlowMetric(BaseMetric):
    """Action-driven synthetic flow vs. video-extracted observed flow.

    Pass ``calibration_path`` at construction to bind the calibration
    once across all samples; otherwise supply ``sample["calibration"]``
    per call. Missing actions or calibration produce a skipped result
    (``score=None``) rather than raising.
    """

    name = "optical_flow.synthetic_optical_flow"
    requires_reference = False
    higher_is_better = False
    needs_gpu = True
    backbone = "optical_flow"
    dependencies = ["ptlflow"]

    def __init__(
        self,
        model_name: str = "dpflow",
        ckpt: str = "things",
        calibration_path: str | Path | None = None,
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
        self._calibration: ThirdPersonCalibration | None = (_resolve_calibration(calibration_path)
                                                            if calibration_path else None)
        self._model = None
        # One frame-pair per DPFlow forward; see ``gt_optical_flow``.
        self._chunk_size = 1

    def to(self, device: str | torch.device) -> SyntheticOpticalFlowMetric:
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

        actions = sample.get("actions")
        if actions is None:
            return self._skip(sample, "missing 'actions' (keyboard + mouse)")

        cal_obj = sample.get("calibration")
        cal = self._calibration if cal_obj is None else _resolve_calibration(cal_obj)
        if cal is None:
            return self._skip(
                sample,
                "missing 'calibration' (pass calibration_path= at construction "
                "or sample['calibration'] per call)",
            )

        video = sample["video"].float()  # (T, C, H, W)
        T, _, H, W = video.shape
        if T < 2:
            raise ValueError("Need at least 2 frames to compute optical flow")
        n_pairs = T - 1

        mouse_pitch_sign = int(sample.get("mouse_pitch_sign", 1))
        chunk = self._chunk_size or 16

        observed = extract_video_flows(
            self._model,
            video,
            chunk=chunk,
            device=self.device,
        )
        predictor = ThirdPersonFlowGenerator(
            calibration=cal,
            frame_shape=(H, W),
            mouse_pitch_sign=mouse_pitch_sign,
        )
        predicted = predictor.generate_flow_sequence(actions, n_pairs=n_pairs)

        n = min(len(observed), len(predicted))
        per_frame = [
            compute_frame_metrics(
                predicted[i],
                observed[i],
                grid_size=self.grid_size,
                min_mag=self.min_mag,
                max_mag_pct=self.max_mag_pct,
            ) for i in range(n)
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
