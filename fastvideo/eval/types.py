from __future__ import annotations

import json
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
class EvalResult:
    """Aggregate output of :meth:`Evaluator.evaluate_dataset`.

    ``summary`` maps metric name → mean score and includes a synthetic
    ``"overall_average"`` key. ``per_video`` is one row per scored video
    with at least ``prompt``, ``video``, and ``scores`` fields.
    """
    summary: dict[str, float]
    per_video: list[dict[str, Any]]

    @classmethod
    def from_raw(cls, by_metric: dict[str, list[float]],
                 per_video: list[dict[str, Any]]) -> EvalResult:
        summary = {m: sum(v) / len(v) for m, v in by_metric.items() if v}
        if summary:
            summary["overall_average"] = sum(summary.values()) / len(summary)
        return cls(summary=summary, per_video=per_video)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"summary": self.summary, "per_video": self.per_video},
            indent=2))

    def print(self) -> None:
        print()
        print("== summary ==")
        for k, v in sorted(self.summary.items()):
            print(f"  {k:42s} {v:.4f}")
