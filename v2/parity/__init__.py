"""Parity plane — a first-class package, not a test folder.

It is how the (recipe, runtime) pair is kept honest: parity is measured by ParityAligner,
the consistency ladder is typed, and the interleave gate is non-negotiable.
"""
from __future__ import annotations

from v2._enums import ConsistencyLevel, ExecutionProfile
from v2.parity.aligner import ParityAligner
from v2.parity.interleave_gate import assert_interleave_parity, compare_outputs
from v2.parity.ladder import Divergence, array_diff, bit_identical, within

__all__ = [
    "ConsistencyLevel", "ExecutionProfile", "ParityAligner", "Divergence", "array_diff", "bit_identical", "within",
    "assert_interleave_parity", "compare_outputs"
]
