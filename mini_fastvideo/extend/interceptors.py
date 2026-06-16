"""cache-dit-style interceptors (design_v3 §11).

``ResidualSkipInterceptor`` is the reference step-skip integration. The load-bearing
correctness property: its per-step state lives in ``LoopState.plugin_state[id][branch]``,
keyed per request AND per CFG branch — NOT a module global. This is exactly why the
interleave gate (§9.3) passes with it on: two interleaved requests have disjoint plugin
state, so neither smears the other's cached prediction.

A 4-step distilled card *rejects* this interceptor (capability negotiation) rather than
producing garbage — handled by InterceptorChain validation + per-card opt-in.
"""
from __future__ import annotations

from typing import Any


class ResidualSkipInterceptor:
    """Skip the model forward on cadence, reusing the previous step's prediction.

    A deliberately simple stand-in for DBCache/FBCache/TaylorSeer: every ``interval``-th
    step is recomputed; intermediate steps reuse the cached output. Demonstrates the
    contract, not the algorithm.
    """
    plugin_id = "residual_skip"
    distribution_altering = True
    graph_safe = False

    def __init__(self, interval: int = 2):
        self.interval = max(2, interval)

    def _branch_state(self, state: Any, branch: str) -> dict:
        ns = state.plugin_state.setdefault(self.plugin_id, {})
        return ns.setdefault(branch, {})

    def before_step(self, plan: Any, state: Any) -> Any | None:
        """Return a cached forward output to skip the model forward on non-cadence steps.

        The step body still runs the cheap solver step with this prediction, so only the
        expensive forward is skipped. Cache lives per request (LoopState.plugin_state) AND
        per branch — interleaving two requests cannot smear caches."""
        branch = str(plan.payload.get("branch", "combined"))
        bs = self._branch_state(state, branch)
        cached = bs.get("last_output")
        if cached is not None and (state.step_idx % self.interval) != 0:
            bs["skipped"] = bs.get("skipped", 0) + 1
            return {"noise_pred": cached}
        return None

    def after_step(self, plan: Any, state: Any, result: Any) -> None:
        branch = str(plan.payload.get("branch", "combined"))
        pred = result.output.get("noise_pred")
        if pred is not None:
            self._branch_state(state, branch)["last_output"] = pred
