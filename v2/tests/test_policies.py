"""Policy semantics: CFG taxonomy, flow-shift, expert routing."""
from __future__ import annotations

import numpy as np

from v2.core.loop.contracts import StepContext
from v2.core.loop.policies import (
    AdaptiveGateCFG,
    BoundaryTimestepRouting,
    ClassicCFG,
    BatchedCFG,
    EmbeddedGuidance,
    FlowShiftPolicy,
    NoRouting,
)


def test_batched_cfg_is_a_dispatch_detail_not_a_mechanism():
    cond, uncond = np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.5, 1.0])
    ctx = StepContext(0, 1.0, 1.0)
    classic = ClassicCFG().combine({"cond": cond, "uncond": uncond}, 5.0, ctx, {})
    batched = BatchedCFG().combine({"cond": cond, "uncond": uncond}, 5.0, ctx, {})
    assert np.array_equal(classic, batched)            # identical output => same policy, two dispatches
    assert BatchedCFG().batched is True and ClassicCFG().batched is False


def test_classic_cfg_formula():
    cond, uncond = np.array([2.0]), np.array([1.0])
    out = ClassicCFG().combine({"cond": cond, "uncond": uncond}, 5.0, StepContext(0, 1, 1), {})
    assert np.allclose(out, uncond + 5.0 * (cond - uncond))


def test_adaptive_gate_reuses_delta_and_self_invalidates_on_expert_switch():
    pol = AdaptiveGateCFG(interval=2)
    cond, uncond, s, state = np.array([1.0, 2.0]), np.array([0.0, 1.0]), 5.0, {}
    c0 = StepContext(0, 1, 1, active_expert_id="e0")           # recompute step
    assert pol.branches_this_step(c0, state) == ["cond", "uncond"]
    out0 = pol.combine({"cond": cond, "uncond": uncond}, s, c0, state)
    assert np.allclose(out0, uncond + s * (cond - uncond))
    c1 = StepContext(1, 1, 1, active_expert_id="e0")           # reuse step (skip uncond forward)
    assert pol.branches_this_step(c1, state) == ["cond"]
    out1 = pol.combine({"cond": cond}, s, c1, state)
    assert np.allclose(out1, cond + (s - 1.0) * (cond - uncond))   # algebraically the same family
    c2 = StepContext(3, 1, 1, active_expert_id="e1")           # expert switch => recompute
    assert pol.branches_this_step(c2, state) == ["cond", "uncond"]


def test_embedded_guidance_is_single_branch_identity_combine():
    pol = EmbeddedGuidance()
    assert pol.branch_vocabulary == ["cond"]
    cond = np.array([3.0, 4.0])
    assert np.array_equal(pol.combine({"cond": cond}, 7.0, StepContext(0, 1, 1), {}), cond)


def test_flow_shift_schedule_monotonic_and_bucketed():
    fs = FlowShiftPolicy(shift=3.0, bucket_lookup={100: 5.0})
    sig = fs.build_schedule(4)
    assert len(sig) == 5
    assert sig[0] == 1.0 and abs(sig[-1]) < 1e-9
    assert all(sig[i] >= sig[i + 1] for i in range(4))
    assert fs.shift_for(10, 10) == 5.0 and fs.shift_for(1, 1) == 3.0


def test_flow_shift_explicit_distilled_sigmas_passthrough():
    fs = FlowShiftPolicy(shift=8.0)
    sig = fs.build_schedule(3, sigmas=[1.0, 0.5, 0.25, 0.0])
    assert list(sig) == [1.0, 0.5, 0.25, 0.0]


def test_expert_routing():
    assert NoRouting("transformer").expert_for(StepContext(0, 1, 0.5)) == "transformer"
    br = BoundaryTimestepRouting("high", "low", boundary=0.5)
    assert br.expert_for(StepContext(0, 1, 0.6)) == "high"
    assert br.expert_for(StepContext(0, 1, 0.4)) == "low"
