"""ParityAligner + consistency ladder (design_v3 §9.1, §9.2)."""
from __future__ import annotations

import numpy as np

from mini_fastvideo.parity import ConsistencyLevel, ParityAligner, array_diff, bit_identical, within


def test_aligner_no_divergence_on_identical_runs():
    ref, cur = ParityAligner(), ParityAligner()
    for s in range(3):
        v = np.ones((2, 2)) * s
        ref.record_tap(s, "latents", v)
        cur.record_tap(s, "latents", v.copy())
    assert cur.first_divergence(ref) is None


def test_aligner_reports_first_divergence_in_step_order():
    ref, cur = ParityAligner(), ParityAligner()
    ref.record_tap(0, "l", np.zeros(2)); cur.record_tap(0, "l", np.zeros(2))
    ref.record_tap(1, "l", np.zeros(2)); cur.record_tap(1, "l", np.ones(2))   # diverges here
    ref.record_tap(2, "l", np.zeros(2)); cur.record_tap(2, "l", np.full(2, 9))
    d = cur.first_divergence(ref)
    assert d is not None and d.where == "l@1"           # first, not the bigger step-2 diff


def test_aligner_respects_per_tap_tolerance():
    ref, cur = ParityAligner(), ParityAligner()
    ref.set_tolerance("l", atol=0.1)
    cur.set_tolerance("l", atol=0.1)
    ref.record_tap(0, "l", np.zeros(2)); cur.record_tap(0, "l", np.full(2, 0.05))
    assert cur.first_divergence(ref) is None             # within tolerance
    cur.record_tap(0, "l", np.full(2, 0.05))             # idempotent re-record


def test_consistency_ladder_ranks_increase():
    assert ConsistencyLevel.C0.rank < ConsistencyLevel.C1.rank < ConsistencyLevel.C2.rank
    assert ConsistencyLevel.C2.rank < ConsistencyLevel.C3.rank < ConsistencyLevel.C4.rank


def test_numeric_helpers():
    assert bit_identical(np.ones(3), np.ones(3))
    assert within(np.ones(3), np.ones(3) + 1e-9, atol=1e-6)
    abs_d, rel_d = array_diff(np.array([1.0]), np.array([2.0]))
    assert abs_d == 1.0 and rel_d == 0.5
