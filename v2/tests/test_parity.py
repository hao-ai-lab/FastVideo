"""ParityAligner + consistency ladder."""
from __future__ import annotations

import numpy as np

from v2.core.parity import ConsistencyLevel, ParityAligner, array_diff, assert_interleave_parity, bit_identical, within
from v2.core.request import DiffusionParams, TaskType, make_request
from v2.recipes import build_default_engine


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


def test_interleave_parity_helper_passes_for_wan_requests():
    eng = build_default_engine()
    reqs = [
        make_request(TaskType.T2V, "wan2.1-1.3b", "a", diffusion=DiffusionParams(num_steps=3, seed=1)),
        make_request(TaskType.T2V, "wan2.1-1.3b", "b", diffusion=DiffusionParams(num_steps=3, seed=2)),
    ]
    assert assert_interleave_parity(eng, reqs) == []
