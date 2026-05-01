"""Verse-Bench AudioBox Aesthetics (CE, CU, PC, PQ).

Uses AudioBox Aesthetics predictor to score audio quality on 4 dimensions.
"""

from __future__ import annotations

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("audio.audiobox_aesthetics")
class VBAudioBoxAestheticsMetric(BaseMetric):
    """Verse-Bench AudioBox Aesthetics: 4 audio quality scores."""

    name = "audio.audiobox_aesthetics"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["audiobox_aesthetics"]

    def __init__(self) -> None:
        super().__init__()
        self._predictor = None

    def to(self, device):
        super().to(device)
        return self

    def setup(self) -> None:
        if self._predictor is not None:
            return
        from audiobox_aesthetics.infer import initialize_predictor
        self._predictor = initialize_predictor()

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        if self._predictor is None:
            self.setup()

        audio = sample["audio"]  # str or list[str]
        if isinstance(audio, str):
            audio = [audio]

        results = []
        for a in audio:
            score = self._predictor.forward([{"path": a}])[0]
            combined = (score['CE'] + score['CU'] + score['PQ'] + (11 - score['PC'])) / 4
            results.append(MetricResult(
                name=self.name,
                score=combined,
                details={
                    "CE": score['CE'],
                    "CU": score['CU'],
                    "PC": score['PC'],
                    "PQ": score['PQ'],
                },
            ))
        return results
