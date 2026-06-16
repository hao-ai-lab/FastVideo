"""Parity plane — a first-class package, not a test folder (design_v3 §9, §18).

It is how the (recipe, runtime) pair is kept honest: parity is measured by ParityAligner,
the consistency ladder is typed, and the interleave gate is non-negotiable.
"""
from __future__ import annotations

from .._enums import ConsistencyLevel, ExecutionProfile
from .aligner import ParityAligner
from .interleave_gate import assert_interleave_parity, compare_outputs
from .ladder import Divergence, array_diff, bit_identical, within

__all__ = ["ConsistencyLevel", "ExecutionProfile", "ParityAligner", "Divergence",
           "array_diff", "bit_identical", "within", "assert_interleave_parity", "compare_outputs"]
