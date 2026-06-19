"""Built-in read-only observers: Profiler, NaNWatch.

ParityAligner lives in its own ``parity/`` package but also implements the Observer ``observe``
protocol so it attaches to the same bus.
"""
from __future__ import annotations

import numpy as np


class Profiler:
    """Per-step wall/CUDA timing that calibrates the cost model.

    Accumulates (work_units, actual_seconds) samples and fits a card's CostModel
    coefficients — the online calibration that refines the conservative baseline.
    """

    def __init__(self) -> None:
        self.samples: list[tuple[str, float, float]] = []  # (batch_key_repr, work_units, seconds)

    def observe(self, event: str, **kw) -> None:
        if event == "step_complete":
            plan, result = kw.get("plan"), kw.get("result")
            if plan is not None and result is not None:
                self.samples.append(
                    (repr(plan.shape_sig.batch_key), float(plan.shape_sig.work_units), float(result.actual_seconds)))

    def calibrate(self, cost_model) -> None:
        """Fit base + per_unit seconds from observed samples (least squares)."""
        if len(self.samples) < 2:
            return
        x = np.array([s[1] for s in self.samples], dtype=np.float64)
        y = np.array([s[2] for s in self.samples], dtype=np.float64)
        if np.ptp(x) == 0:
            cost_model.base_seconds = float(y.mean())
            return
        slope, intercept = np.polyfit(x, y, 1)
        cost_model.per_unit_seconds = float(max(slope, 0.0))
        cost_model.base_seconds = float(max(intercept, 0.0))


class NaNWatch:
    """First-NaN/Inf localization. Request-fatal, triggering an SPMD-consistent abort."""

    def __init__(self) -> None:
        self.first: tuple[str, str] | None = None  # (tap/output name, plan label)

    def observe(self, event: str, **kw) -> None:
        if event == "step_complete" and self.first is None:
            plan, result = kw.get("plan"), kw.get("result")
            if result is None:
                return
            for name, val in result.output.items():
                arr = np.asarray(val) if hasattr(val, "__array__") or isinstance(val, list | tuple) else None
                if arr is not None and arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
                    self.first = (name, getattr(plan, "label", "") or name)
                    return

    @property
    def tripped(self) -> bool:
        return self.first is not None
