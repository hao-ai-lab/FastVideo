"""CacheDiTDenoiseLoop — the loop OWNS content-adaptive control flow (design_v3 §2.2).

The signature payoff of loop inversion: because the *model* owns control flow, content-adaptive
decisions are natural. This loop, at each step, decides whether to:

  * **skip the expensive forward** and reuse the cached velocity when consecutive predictions barely
    change (cache-dit / TeaCache / Δ-DiT — a 1.5–2× inference win), or
  * **early-exit** when the latent has converged.

The result is a *variable, content-dependent* step count — impossible in a ``for t in timesteps``
runtime, trivial in a driven loop whose ``next()`` reads ``LoopState``. It is fully isolated from the
shared ``WanDenoiseLoop`` (subclass, threshold 0 ⇒ identical behavior), and still interleave-safe:
different requests skip different steps, and the parity gate still holds (ragged loops don't smear).
"""
from __future__ import annotations

import numpy as np

from v2.loop.contracts import Done, StepResult
from v2.platform import FLOW_MATCH_STEP
from v2.recipes.wan21.loop import WanDenoiseLoop


class CacheDiTDenoiseLoop(WanDenoiseLoop):
    def __init__(self, *, cache_threshold: float = 0.0, exit_threshold: float = 0.0, **kw):
        super().__init__(**kw)
        self.cache_threshold = cache_threshold     # rel velocity change below which we reuse (skip the DiT)
        self.exit_threshold = exit_threshold       # rel latent change below which we stop early

    def init(self, req, model, ctx):
        st = super().init(req, model, ctx)
        st.scratch.update(last_v=None, skip_next=False, skipped=0, exited=False)
        return st

    def next(self, st):
        if st.scratch.get("exited"):
            return Done()                          # converged early — content-adaptive termination
        plan = super().next(st)                    # the normal full-forward step plan
        last_v = st.scratch.get("last_v")
        if isinstance(plan, Done) or not self.cache_threshold or not st.scratch.get("skip_next") \
                or last_v is None:
            return plan
        # SKIP: reuse the cached velocity, no DiT forward (the cache-dit win) — wrap the run thunk
        x, prec = st.latents["video"], self.precision
        sigma_t, sigma_next = st.sigmas[st.step_idx], st.sigmas[st.step_idx + 1]
        v = np.asarray(last_v, dtype="float32")

        def run(model, override=None):
            x_next = model.platform.kernels.get(FLOW_MATCH_STEP)(prec.cast(x), v, sigma_t, sigma_next)
            return StepResult(output={"noise_pred": v, "latents": x_next.astype("float32"), "skipped": True})

        plan.run = run
        plan.label = f"{plan.label}.skip"
        return plan

    def advance(self, st, result):
        x_prev = np.asarray(st.latents["video"], dtype="float64")
        v = np.asarray(result.output["noise_pred"], dtype="float64")
        prev_v = st.scratch.get("last_v")
        st = super().advance(st, result)           # folds latents, step_idx, (rollout) trajectory
        if result.output.get("skipped"):
            st.scratch["skipped"] += 1
            st.scratch["skip_next"] = False         # don't chain skips — re-check with a full forward
        elif self.cache_threshold and prev_v is not None:
            rel = float(np.linalg.norm(v - np.asarray(prev_v, dtype="float64"))
                        / (np.linalg.norm(prev_v) + 1e-8))
            st.scratch["skip_next"] = rel < self.cache_threshold   # next prediction will barely change
        st.scratch["last_v"] = result.output["noise_pred"]
        half = max(1, (len(st.sigmas) - 1) // 2)        # only consider exiting in the low-noise tail
        if self.exit_threshold and st.step_idx >= half:
            x_next = np.asarray(st.latents["video"], dtype="float64")
            dl = float(np.linalg.norm(x_next - x_prev) / (np.linalg.norm(x_prev) + 1e-8))
            if dl < self.exit_threshold:
                st.scratch["exited"] = True         # latent converged → stop next tick
        return st

    def finalize(self, st):
        res = super().finalize(st)
        res.metrics["skipped_steps"] = float(st.scratch.get("skipped", 0))
        res.metrics["early_exited"] = 1.0 if st.scratch.get("exited") else 0.0
        return res
