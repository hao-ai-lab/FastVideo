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
