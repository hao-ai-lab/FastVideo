"""The consistency ladder + numeric comparison helpers.

C0 component · C1 loop · C2 behavioral · C3 distribution · C4 artifact.

The C2 split is load-bearing: likelihood-based methods compare per-step log-probs;
likelihood-free methods (DiffusionNFT) compare seeded final-sample + prediction-space
identity (old_deviate / ref-MSE) — there are NO log-probs to match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from v2._enums import ConsistencyLevel  # re-exported for the package


@dataclass
class Divergence:
    where: str  # tap name / artifact path
    level: ConsistencyLevel
    max_abs_diff: float
    max_rel_diff: float
    message: str = ""

    def __bool__(self) -> bool:
        return True


def array_diff(a: Any, b: Any) -> tuple[float, float]:
    """Return (max_abs, max_rel) difference between two array-likes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return (float("inf"), float("inf"))
    if a.size == 0:
        return (0.0, 0.0)
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, diff / denom, 0.0)
    return (max_abs, float(np.nanmax(rel)) if rel.size else 0.0)


def within(a: Any, b: Any, rtol: float = 0.0, atol: float = 0.0) -> bool:
    """Close enough if within the absolute OR the relative tolerance (allclose-style).
    With atol=rtol=0 this reduces to exact equality (the bit-identical case)."""
    abs_d, rel_d = array_diff(a, b)
    return abs_d <= atol or rel_d <= rtol


def bit_identical(a: Any, b: Any) -> bool:
    """C1/C2 bit-identical check (fixed seed, same kernels) — array_equal on raw values."""
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except Exception:
        return a == b
