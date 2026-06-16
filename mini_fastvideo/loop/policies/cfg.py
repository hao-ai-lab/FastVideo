"""CFGPolicy — one taxonomy over one shared denoise body (design_v3 §5.3).

> Batched-vs-two-forward is a *dispatch detail inside one policy*, not a separate mechanism.

The step body asks ``branches_this_step(ctx, state)`` which forwards to run, runs them, then
calls ``combine(preds, scale, ctx, state)``. Per-request mutable state (the adaptive-gate
cached delta) lives in the ``state`` dict, which the step body slices out of
``LoopState.plugin_state`` — never a module global (the §5.1 safety property).

cfg-parallel is a *parallelism axis* (not a policy); companions are an *orchestrator pattern*
(not in the loop). Both compose with any policy here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..contracts import StepContext


class CFGPolicy(ABC):
    batched: bool = False                     # dispatch detail: stack branches into one forward
    branch_vocabulary: list[str] = ["cond", "uncond"]   # full declared set (for per-branch plugin state)

    def branches_this_step(self, ctx: StepContext, state: dict) -> list[str]:
        return list(self.branch_vocabulary)

    @abstractmethod
    def combine(self, preds: dict[str, np.ndarray], guidance_scale: float,
                ctx: StepContext, state: dict) -> np.ndarray: ...


class ClassicCFG(CFGPolicy):
    """Sequential two-forward CFG: ``uncond + s·(cond − uncond)`` (Wan default)."""

    def combine(self, preds, guidance_scale, ctx, state):
        cond, uncond = preds["cond"], preds["uncond"]
        return uncond + guidance_scale * (cond - uncond)


class BatchedCFG(ClassicCFG):
    """Same algebra; the two branches are stacked into one batched forward (dispatch detail).

    Identical output to ClassicCFG — verified by a parity test — which is the whole point:
    batched-vs-2-forward is not a separate mechanism.
    """
    batched = True


class EmbeddedGuidance(CFGPolicy):
    """Degenerate single-branch identity-combine (Flux): guidance rides in the forward kwarg.

    This is *not* 'no CFG' — it is kept inside the same abstraction (design_v3 §5.3)."""
    branch_vocabulary = ["cond"]

    def combine(self, preds, guidance_scale, ctx, state):
        return preds["cond"]


class AdaptiveGateCFG(CFGPolicy):
    """Cached-delta reuse with expert-switch self-invalidation (design_v3 §5.1 canonical state).

    On reuse steps it runs ONLY the cond branch and reuses the cached delta:
    ``out = cond + (s−1)·delta`` (algebraically identical to ``uncond + s·(cond−uncond)``).
    The cached delta is invalidated when ``ExpertRouting`` switches the active expert.
    """

    def __init__(self, interval: int = 2):
        self.interval = max(1, interval)

    def _recompute(self, ctx, state) -> bool:
        return (ctx.step_idx % self.interval == 0
                or "delta" not in state
                or state.get("expert") != ctx.active_expert_id)

    def branches_this_step(self, ctx, state):
        return ["cond", "uncond"] if self._recompute(ctx, state) else ["cond"]

    def combine(self, preds, guidance_scale, ctx, state):
        if "uncond" in preds:                       # recompute path
            delta = preds["cond"] - preds["uncond"]
            state["delta"] = delta
            state["expert"] = ctx.active_expert_id
            return preds["uncond"] + guidance_scale * delta
        delta = state["delta"]                       # reuse path (skipped the uncond forward)
        return preds["cond"] + (guidance_scale - 1.0) * delta


class PerModalityCFG(CFGPolicy):
    """Joint A/V per-modality scales + interval gating (LTX-2, Cosmos3 t2vs — phase 2).

    ``combine`` reads the active modality from ``ctx.extra['modality']`` and applies its scale.
    """

    def __init__(self, scales: dict[str, float] | None = None, interval_gated: bool = False):
        self.scales = scales or {"video": 5.0}
        self.interval_gated = interval_gated

    def combine(self, preds, guidance_scale, ctx, state):
        modality = ctx.extra.get("modality", "video")
        scale = self.scales.get(modality, guidance_scale)
        cond, uncond = preds["cond"], preds["uncond"]
        return uncond + scale * (cond - uncond)
