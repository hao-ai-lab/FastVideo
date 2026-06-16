"""LTX2DenoiseLoop — a custom-step driven loop (design_v3 §5.1 escape hatch; §15 LTX-2 2-stage).

LTX-2's guidance is braided (multi-pass: positive / negative / STG-perturbed, combined per the
LTX formula), so the step body is hand-written using samplers + the DiT as a library — exactly
the legitimate custom-step pattern. It is still a Loop (scheduled, admitted, observed, parity-
gated). One ``LTX2DenoiseLoop`` is instantiated per stage (base / refine) with that stage's
distilled sigma schedule; both stages bind the SAME ``transformer`` component (shared by
reference — the MoT 'one instance, many loops' property, demonstrated even here).

Real LTX-2 distilled schedules (from the repo):
  base  : [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]   (8 steps)
  refine: [0.909375, 0.725, 0.421875, 0.0]                                          (3 steps)
The real DiT predicts ``denoised`` (x0); the CPU toy predicts velocity, integrated by the same
flow-match Euler step — the 2-stage structure, distilled schedules, and multi-pass guidance are
what this models faithfully.
"""
from __future__ import annotations

import numpy as np

from ..._enums import ExecutionProfile, WorkUnitKind
from ...loop.contracts import (
    Done,
    LoopResult,
    LoopState,
    ResourceRequest,
    ShapeSignature,
    StepContext,
    StepResult,
    WorkPlan,
)
from ...loop.sampler import flow_match_euler_step
from ..backend import LATENT_CHANNELS

BASE_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
REFINE_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


def ltx_base_latent_shape(req) -> tuple[int, int, int, int]:
    d = req.diffusion
    t = max(1, d.num_frames // 40)
    # refine halves requested resolution for stage-1; stage-2 upsamples 2× (design §15)
    h = max(2, d.height // 240)
    w = max(2, d.width // 240)
    return (LATENT_CHANNELS, t, h, w)


class LTX2DenoiseLoop:
    def __init__(self, *, loop_id, stage, sigmas, cfg_scale, stg_scale, cost,
                 input_slot=None, seed_offset=0):
        self.loop_id = loop_id
        self.stage = stage                 # "base" | "refine"
        self.sigmas = list(sigmas)
        self.cfg_scale = cfg_scale
        self.stg_scale = stg_scale
        self.cost = cost
        self.input_slot = input_slot       # None => fresh noise (base); slot name => read latents (refine)
        self.seed_offset = seed_offset

    def init(self, req, model, ctx) -> LoopState:
        seed = (req.diffusion.seed if req.diffusion.seed is not None else 0) + self.seed_offset
        rng = np.random.default_rng(seed)
        sig = self.sigmas
        if self.input_slot is None:
            x = (rng.standard_normal(ltx_base_latent_shape(req)) * float(sig[0])).astype("float32")
        else:
            x = np.asarray(ctx.slots[self.input_slot], dtype="float32")
        st = LoopState(loop_id=self.loop_id, instance_id=model.card.model_id,
                       request_id=req.request_id, profile=ctx.profile, rng=rng, seed=seed,
                       latents={"video": x}, sigmas=[float(s) for s in sig],
                       timesteps=[float(s) * 1000.0 for s in sig])
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        x = st.latents["video"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        cfg_scale, stg_scale = self.cfg_scale, self.stg_scale

        def run(model, override=None):
            if override is not None and "noise_pred" in override:
                velocity = np.asarray(override["noise_pred"], dtype="float32")
            else:
                dit = model.component("transformer")
                pos = dit(x, pe, sigma_t)                       # full conditioned pass
                neg = dit(x, ne, sigma_t)                       # negative (CFG) pass
                # STG-perturbed pass (here: drop text conditioning) — the multi-pass braid
                ptb = dit(x, np.zeros_like(pe) if pe is not None else None, sigma_t)
                velocity = (pos
                            + (cfg_scale - 1.0) * (pos - neg)
                            + stg_scale * (pos - ptb))
            x_next = flow_match_euler_step(x, np.asarray(velocity, dtype="float32"), sigma_t, sigma_next)
            return StepResult(output={"noise_pred": np.asarray(velocity, dtype="float32"),
                                      "latents": x_next.astype("float32")})

        res = ResourceRequest(
            compute_seconds=self.cost.predict(int(np.prod(x.shape)), 3.0),  # 3 passes
            resident_bytes=int(x.nbytes), peak_activation_bytes=int(x.nbytes * 3))
        return WorkPlan(
            loop_id=self.loop_id, instance_id=st.instance_id, kind=WorkUnitKind.DIFFUSION_STEP,
            shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP, dims=tuple(x.shape),
                                     extra=(("stage", self.stage),)),
            resources=res, payload={"branch": "combined", "step": i, "stage": self.stage}, run=run,
            label=f"ltx2.{self.stage}.{i}")

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["video"] = result.output["latents"]
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({"step": st.step_idx, "stage": self.stage,
                                  "latents": np.asarray(st.latents["video"]).copy()})
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["video"]},
                          metrics={f"{self.stage}_steps": float(st.step_idx)},
                          behavior=st.trajectory or None)
