"""Built-in read-only observer: NaNWatch.

ParityAligner lives in its own ``parity/`` package but also implements the Observer ``observe``
protocol so it attaches to the same bus.
"""
from __future__ import annotations

import numpy as np


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
