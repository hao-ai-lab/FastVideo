"""Parity plane — a first-class package, not a test folder.

It is how the (recipe, runtime) pair is kept honest: parity is measured by ParityAligner,
the consistency ladder is typed, and the interleave gate is non-negotiable.
"""
from __future__ import annotations

from v2.core.enums import ConsistencyLevel, ExecutionProfile
from v2.core.parity.aligner import ParityAligner
from v2.core.parity.compare import compare_outputs
from v2.core.parity.ladder import Divergence, array_diff, bit_identical, within

__all__ = [
    "ConsistencyLevel", "ExecutionProfile", "ParityAligner", "Divergence", "array_diff", "bit_identical", "within",
    "compare_outputs"
]
