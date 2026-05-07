from __future__ import annotations

from dataclasses import dataclass, field
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
