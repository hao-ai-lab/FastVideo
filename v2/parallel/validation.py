"""Pre-flight parallelism validation (design_v3 §8).

> Pre-flight or it fails at load, never halfway. Ownership conflicts are build errors
> ... Applicability conditions travel with axes: ``pp_patch`` is invalid for causal/AR.

All checks are CPU-testable on a fake mesh (design.md §6.3.4: CPU-testability).
"""
from __future__ import annotations

from .._enums import LoopKind
from .plan import AXIS_NAMES, ParallelPlan


class ParallelValidationError(ValueError):
    pass


# loop kinds whose causality is broken by PipeFusion displaced-patch pipelining
_CAUSAL_KINDS = {LoopKind.CHUNK_ROLLOUT, LoopKind.AR_DECODE}


def validate_plan(plan: ParallelPlan, card=None, *, world_size: int | None = None,
                  cfg_policy_batched: bool = False) -> ParallelPlan:
    """Validate a plan, optionally against a card. Returns the plan or raises.

    Checks (design_v3 §8):
      1. axis names are known; degrees >= 1
      2. cfgp <= 2
      3. product of degrees matches world_size (when given)
      4. pp_patch invalid for any causal/AR loop on the card
      5. ownership conflict: cfgp>1 AND a batched CFGPolicy is rejected (§5.3, §9 build-guard)
    """
    errs: list[str] = []

    for name, deg in plan.axes.items():
        if name not in AXIS_NAMES:
            errs.append(f"unknown parallel axis {name!r} (known: {AXIS_NAMES})")
        if int(deg) < 1:
            errs.append(f"axis {name!r} has degree {deg} < 1")

    if plan.degree("cfgp") > 2:
        errs.append(f"cfgp degree {plan.degree('cfgp')} > 2 (CFG has at most 2 branches)")

    if world_size is not None and plan.world_size() != world_size:
        errs.append(f"product of degrees {plan.world_size()} != world_size {world_size}")

    if plan.degree("cfgp") > 1 and cfg_policy_batched:
        errs.append("ownership conflict: a request owns a BatchedCFG policy OR a cfgp group, never both "
                    "(design_v3 §5.3 / §9 build-guard)")

    if card is not None and plan.degree("pp_patch") > 1:
        causal = [lid for lid, lp in card.loops.items() if lp.kind in _CAUSAL_KINDS]
        if causal:
            errs.append(f"pp_patch is invalid for causal/AR loops {causal} (stale KV breaks causality, §8)")

    if errs:
        raise ParallelValidationError("ParallelPlan failed validation:\n  - " + "\n  - ".join(errs))
    return plan
