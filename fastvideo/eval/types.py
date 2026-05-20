from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MetricResult:
    """Standard result container returned by all metrics.

    ``score`` is ``None`` when the metric was skipped (e.g. missing
    required input).  Check ``details["skipped"]`` for the reason.
    """
    name: str
    score: float | None
    details: dict[str, Any] = field(default_factory=dict)


class EvalResults(list):
    """Return type for :meth:`Evaluator.evaluate` with ``samples=...``.

    Behaves like a ``list[dict[str, MetricResult]]`` — one dict per
    input sample, in input order — so existing iteration and indexing
    keeps working. The ``corpus`` attribute carries set-metric results
    (FAD, IS, …) that are properties of the whole input set, not of
    any individual sample. Empty dict when no set metric ran.
    """

    def __init__(
        self,
        samples: list[dict[str, MetricResult]] | None = None,
        corpus: dict[str, MetricResult] | None = None,
    ) -> None:
        super().__init__(samples or [])
        self.corpus: dict[str, MetricResult] = corpus or {}


@dataclass
class Video:
    """Path-backed media handle. The :class:`VideoPool` populates
    ``frames`` (and optionally ``audio``) before the metric loop sees
    the sample.
    """

    source: Any
    fps: float | None = None
    frames: Any = None
    audio: Any = None
    audio_sr: int | None = None

    def has_frames(self) -> bool:
        return self.frames is not None

    def has_audio(self) -> bool:
        return self.audio is not None

    def __post_init__(self) -> None:
        if isinstance(self.source, Path):
            self.source = str(self.source)
