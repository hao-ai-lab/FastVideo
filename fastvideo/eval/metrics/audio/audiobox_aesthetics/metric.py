"""AudioBox Aesthetics (CE, CU, PC, PQ).

Thin wrapper around Meta's ``audiobox_aesthetics`` predictor. Returns
four per-clip dimensions (CE — Content Enjoyment, CU — Content
Usefulness, PC — Production Complexity, PQ — Production Quality);
``score`` exposes PQ, the dimension V2A papers typically report on.
The remaining three are surfaced under ``details``.

The earlier Verse-Bench combined score ``(CE + CU + PQ + (11 − PC)) / 4``
is non-standard and is deliberately not used.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("audio.audiobox_aesthetics")
class AudioBoxAestheticsMetric(BaseMetric):
    """AudioBox Aesthetics: PQ as the primary score, CE/CU/PC/PQ in details."""

    name = "audio.audiobox_aesthetics"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    is_set_metric = False
    dependencies = ["audiobox_aesthetics"]

    def __init__(self) -> None:
        super().__init__()
        self._predictor: Any = None

    def to(self, device):
        super().to(device)
        if self._predictor is not None:
            self._predictor.model.to(self.device)
            self._predictor.device = self.device
        return self

    def setup(self) -> None:
        if self._predictor is not None:
            return
        from audiobox_aesthetics.infer import initialize_predictor
        predictor = initialize_predictor()
        # Upstream's setup_model() always lands on default CUDA (cuda:0).
        # Re-pin onto this worker's device so multi-GPU eval actually parallelizes.
        predictor.model.to(self.device)
        predictor.device = self.device
        self._predictor = predictor

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        from pathlib import Path

        if self._predictor is None:
            self.setup()

        audio_path = sample.get("audio")
        if audio_path is None:
            return self._skip(sample, "missing 'audio'")
        if not Path(audio_path).exists():
            return self._skip(sample, f"audio file not found: {audio_path}")

        try:
            score = self._predictor.forward([{"path": audio_path}])[0]
        except Exception as exc:  # pragma: no cover — upstream raises vary
            return self._skip(sample, f"audiobox predictor failed: {type(exc).__name__}: {exc}")
        try:
            return MetricResult(
                name=self.name,
                score=float(score["PQ"]),
                details={
                    "CE": float(score["CE"]),
                    "CU": float(score["CU"]),
                    "PC": float(score["PC"]),
                    "PQ": float(score["PQ"]),
                },
            )
        except (KeyError, TypeError) as exc:
            return self._skip(sample, f"audiobox returned unexpected shape: {exc}")
