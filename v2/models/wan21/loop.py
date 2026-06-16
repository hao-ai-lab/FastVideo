"""WanDenoiseLoop — the canonical driven denoise loop (design_v3 §5, §15a).

Bidirectional video diffusion: N flow-match steps over the full latent. Composed from policies
(CFG / flow-shift / precision / expert routing) — the default decomposition. ``next`` is
kernel-free (it builds the forward+combine+solver thunk); ``advance`` folds the result and,
under the ROLLOUT profile, captures a behavior slice — so the *same* loop serves and rolls out
for RL (design_v3 §10). All per-request state lives in ``LoopState`` (interleave-safe).
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
from ...loop.sampler import flow_match_euler_step, flow_sde_step_with_logprob
from ...request.streams import StreamChunk
from ..backend import LATENT_CHANNELS


def latent_shape(req) -> tuple[int, int, int, int]:
    """Map request geometry to a small (deterministic) latent for the CPU toy backend."""
    d = req.diffusion
    t = max(1, d.num_frames // 40)
    h = max(2, d.height // 120)
    w = max(2, d.width // 120)
    return (LATENT_CHANNELS, t, h, w)


class WanDenoiseLoop:
    def __init__(self, *, loop_id, cfg, flow_shift, precision, expert, cost):
        self.loop_id = loop_id
        self.cfg = cfg
        self.flow_shift = flow_shift
        self.precision = precision
        self.expert = expert
        self.cost = cost

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        sig = self.flow_shift.build_schedule(req.diffusion.num_steps, req.diffusion.height,
                                             req.diffusion.width, sigmas=req.diffusion.sigmas or None)
        shape = latent_shape(req)
        x = (rng.standard_normal(shape) * float(sig[0])).astype("float32")
        st = LoopState(loop_id=self.loop_id, instance_id=model.card.model_id,
                       request_id=req.request_id, profile=ctx.profile, rng=rng, seed=seed,
                       latents={"video": x}, sigmas=[float(s) for s in sig],
                       timesteps=[float(s) * 1000.0 for s in sig])
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        # FlowGRPO RL rollout: switch the sampler to SDE-with-logprob (else deterministic ODE serve).
        st.scratch["sde"] = bool(getattr(req.diffusion, "sde_rollout", False))
        st.scratch["sde_noise_scale"] = float(getattr(req.diffusion, "sde_noise_scale", 0.7))
        st.plugin_state["cfg"] = {}
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        expert_id = self.expert.expert_for(StepContext(i, st.timesteps[i], sigma_t))
        sctx = StepContext(step_idx=i, timestep=st.timesteps[i], sigma=sigma_t, active_expert_id=expert_id)
        cfg_state = st.plugin_state["cfg"]
        branches = self.cfg.branches_this_step(sctx, cfg_state)
        x = st.latents["video"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision
        sde, noise_scale, rng = st.scratch.get("sde", False), st.scratch.get("sde_noise_scale", 0.7), st.rng

        def run(model, override=None):
            if override is not None and "noise_pred" in override:
                velocity = np.asarray(override["noise_pred"], dtype="float32")
            else:
                dit = model.component(expert_id)
                preds = {b: dit(x, pe if b == "cond" else ne, sigma_t) for b in branches}
                velocity = cfg.combine(preds, scale, sctx, cfg_state)
            velocity = precision.cast(velocity)
            if sde:                                              # FlowGRPO rollout: stochastic + log-prob
                noise = rng.standard_normal(x.shape)
                x_next, logp, _m, _s = flow_sde_step_with_logprob(
                    precision.cast(x), velocity, sigma_t, sigma_next, noise=noise, noise_scale=noise_scale)
                return StepResult(output={"noise_pred": np.asarray(velocity, dtype="float32"),
                                          "latents": x_next.astype("float32"),
                                          "sde_logprob": logp, "prev": np.asarray(x, dtype="float32")})
            x_next = flow_match_euler_step(precision.cast(x), velocity, sigma_t, sigma_next)
            return StepResult(output={"noise_pred": np.asarray(velocity, dtype="float32"),
                                      "latents": x_next.astype("float32")})

        cond_bytes = sum(int(np.asarray(e).nbytes) for e in (pe, ne) if e is not None)
        res = ResourceRequest(
            compute_seconds=self.cost.predict(int(np.prod(x.shape)), float(len(branches))),
            resident_bytes=int(x.nbytes) + cond_bytes,    # latents + conditioning held for the loop
            peak_activation_bytes=int(x.nbytes))          # one step's transient working buffer
        emits = []
        if st.scratch.get("stream_video"):
            emits.append(StreamChunk(stream_id=st.request_id, modality="video", seq=i,
                                     data=x, preview=True))   # carry the latent as a preview payload
        return WorkPlan(
            loop_id=self.loop_id, instance_id=st.instance_id, kind=WorkUnitKind.DIFFUSION_STEP,
            shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP, dims=tuple(x.shape),
                                     extra=(("cfg", type(cfg).__name__),)),
            resources=res, payload={"branch": "combined", "step": i}, run=run,
            label=f"wan.denoise.{i}", emits=emits)

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["video"] = result.output["latents"]
        if st.profile == ExecutionProfile.ROLLOUT:
            i = st.step_idx
            rec = {"step": i, "sigma": st.sigmas[i],
                   "velocity": np.asarray(result.output["noise_pred"]).copy(),
                   "latents": np.asarray(st.latents["video"]).copy()}
            if "sde_logprob" in result.output:                   # FlowGRPO: capture the PPO log-prob slice
                rec.update(sde_logprob=result.output["sde_logprob"],
                           prev=np.asarray(result.output["prev"]).copy(),
                           sample=np.asarray(result.output["latents"]).copy(),
                           sigma_t=st.sigmas[i], sigma_next=st.sigmas[i + 1])
            st.trajectory.append(rec)
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["video"]},
                          metrics={"denoise_steps": float(st.step_idx)},
                          behavior=st.trajectory or None)
