"""TurboWanDenoiseLoop — the rCM (Reparameterized Consistency Model) few-step denoise loop.

TurboWan reuses the Wan architecture but swaps the multistep flow-match sampler for the rCM consistency
sampler, so this loop is the only genuinely new piece vs ``v2/recipes/wan21``: the FLOW_MATCH schedule +
Euler step become the TrigFlow->RectifiedFlow schedule + stochastic consistency step in ``sampler.py``
(a faithful port of ``RCMScheduler``).

Per step (mirroring ``fastvideo/pipelines/stages/denoising.py``):
  * the model timestep is the scaled sigma ``t = sigma * 1000`` (the DiT's training convention);
  * the DiT predicts a velocity (CFG-combined when guidance_scale != 1 and a negative branch exists — the
    presets use guidance_scale 1.0, so CFG is off);
  * the rCM SDE step ``x_next = (1 - t_next)*(x - t_cur*v) + t_next*noise`` injects fresh per-step noise.
    The host RNG makes every step eager (``capturable=False``), like the cosmos2 EDM and FlowGRPO SDE loops.

MoE + i2v (Wan2.2-I2V-A14B): expert routing and i2v conditioning hooks are reused unchanged from the Wan
loop. ``self.expert`` (a ``BoundaryTimestepRouting`` for the MoE variant) picks the high/low-noise transformer
per step by comparing the raw sigma against the boundary (fastvideo's ``0.9*num_train_timesteps`` vs the
scaled timestep ``sigma*1000``). The i2v ``[mask|cond]`` latent + CLIP image embeds flow from slots to the
WanDiT adapter as in ``v2/recipes/wan21/i2v.py``; ``None`` for T2V -> the plain forward.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2._enums import ExecutionProfile, WorkUnitKind
from v2.loop.contracts import (
    Done,
    LoopResult,
    LoopState,
    ResourceRequest,
    ShapeSignature,
    StepContext,
    StepResult,
    WorkPlan,
)
from v2.recipes.turbowan.sampler import build_rcm_sigmas
from v2.recipes.turbowan.sampler import rcm_step as _rcm_step
from v2.recipes.wan21.loop import (
    WAN_LATENT_CHANNELS,
    WAN_SPATIAL_RATIO,
    WAN_TEMPORAL_RATIO,
    latent_shape,
)


class TurboWanDenoiseLoop:
    """rCM few-step denoise over the full Wan latent. Same lifecycle/contract as ``WanDenoiseLoop``."""

    def __init__(self,
                 *,
                 loop_id: str,
                 cfg: Any,
                 precision: Any,
                 expert: Any,
                 cost: Any,
                 sigma_max: float = 80.0,
                 latent_channels: int = WAN_LATENT_CHANNELS,
                 spatial_ratio: int = WAN_SPATIAL_RATIO,
                 temporal_ratio: int = WAN_TEMPORAL_RATIO):
        self.loop_id = loop_id
        self.cfg = cfg  # carried for the WorkPlan op-structure key; few-step recipe is CFG-off
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.sigma_max = sigma_max
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio

    def init(self, req: Any, model: Any, ctx: Any) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        sig = build_rcm_sigmas(req.diffusion.num_steps, sigma_max=self.sigma_max)
        shape = latent_shape(req,
                             model,
                             channels=self.latent_channels,
                             spatial_ratio=self.spatial_ratio,
                             temporal_ratio=self.temporal_ratio)
        # rCM ``scale_noise``/``init_noise_sigma``: the initial latent is randn scaled by the first sigma.
        x = (rng.standard_normal(shape) * float(sig[0])).astype("float32")
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"video": x},
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) * 1000.0 for s in sig])  # model timestep = sigma*1000
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        # I2V (Wan2.2-I2V-A14B): the program writes the CLIP image embeds + the [mask|cond] latent. Absent
        # for T2V -> None -> the plain Wan forward (same hooks as v2/recipes/wan21/i2v.py).
        st.scratch["i2v_cond"] = ctx.slots.get("i2v_cond")
        st.scratch["i2v_img_embeds"] = ctx.slots.get("i2v_img_embeds")
        return st

    def next(self, st: LoopState) -> WorkPlan | Done:
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        t = st.timesteps[i]  # sigma*1000, the model timestep
        expert_id = self.expert.expert_for(StepContext(i, t, sigma_t))
        x = st.latents["video"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        precision = self.precision
        do_cfg = scale != 1.0 and ne is not None  # few-step TurboWan preset: scale==1 -> off
        rng = st.rng
        i2v_ctx, i2v_cond = st.scratch.get("i2v_img_embeds"), st.scratch.get("i2v_cond")

        def _velocity(model: Any, dit: Any) -> np.ndarray:
            # The DiT velocity at the SCALED timestep ``t``, with optional CFG. i2v_ctx/i2v_cond are None
            # for T2V (plain forward); for i2v the Wan adapter concats the cond latent + passes CLIP embeds.
            cond_v = np.asarray(dit(x, pe, t, context=i2v_ctx, cond=i2v_cond), dtype="float32")
            if do_cfg:
                uncond_v = np.asarray(dit(x, ne, t, context=i2v_ctx, cond=i2v_cond), dtype="float32")
                cond_v = uncond_v + scale * (cond_v - uncond_v)
            return precision.cast(cond_v)

        def run(model: Any, override: Any = None) -> StepResult:
            dit = model.component(expert_id)
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                velocity = _velocity(model, dit)
            # rCM SDE consistency step with FRESH per-step noise (the consistency sampler, not flow-match).
            noise = rng.standard_normal(x.shape)
            x_next = _rcm_step(precision.cast(x), velocity, sigma_t, sigma_next, noise)
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": x_next.astype("float32")
            })

        cond_bytes = sum(int(np.asarray(e).nbytes) for e in (pe, ne) if e is not None)
        res = ResourceRequest(compute_seconds=self.cost.predict(int(np.prod(x.shape)), 2.0 if do_cfg else 1.0),
                              resident_bytes=int(x.nbytes) + cond_bytes,
                              peak_activation_bytes=int(x.nbytes))
        emits = []
        if st.scratch.get("stream_video"):
            from v2.request.streams import StreamChunk
            emits.append(StreamChunk(stream_id=st.request_id, modality="video", seq=i, data=x, preview=True))
        return WorkPlan(
            loop_id=self.loop_id,
            instance_id=st.instance_id,
            kind=WorkUnitKind.DIFFUSION_STEP,
            shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                     dims=tuple(x.shape),
                                     dtype=precision.compute_dtype,
                                     extra=(("cfg", type(self.cfg).__name__), ("rcm", True))),
            resources=res,
            payload={
                "branch": "rcm",
                "step": i
            },
            run=run,
            label=f"turbowan.denoise.{i}",
            emits=emits,
            # The rCM step draws fresh host RNG (stochastic consistency step) -> eager-break, never captured.
            capturable=False)

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["video"] = result.output["latents"]
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({
                "step": st.step_idx,
                "sigma": st.sigmas[st.step_idx],
                "velocity": np.asarray(result.output["noise_pred"]).copy(),
                "latents": np.asarray(st.latents["video"]).copy()
            })
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["video"]},
                          metrics={"denoise_steps": float(st.step_idx)},
                          behavior=st.trajectory or None)
