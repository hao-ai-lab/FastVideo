"""LongCatDenoiseLoop — a faithful fork of ``WanDenoiseLoop`` for LongCat-Video T2V.

LongCat is flow-match (deterministic Euler over a ``FlowMatchEulerDiscreteScheduler``), but it differs from
Wan in three load-bearing ways (faithful to ``fastvideo/pipelines/stages/longcat_denoising.py`` +
``longcat_refine_timestep.py``):

  1. SIGMA SCHEDULE (spec blocker #2): NOT ``build_flow_sigmas`` / flow-shift (``flow_shift=None``). LongCat's
     ``get_timesteps_sigmas`` (non-distill) uses ``torch.linspace(1.0, 0.001, num_inference_steps)`` — exactly
     ``num_steps`` points terminating at 0.001 (NOT 0) — and the scheduler appends a trailing ``0.0`` for the
     final boundary. ``FlowShiftPolicy.build_schedule`` would give ``num_steps+1`` points terminating at 0.0
     (wrong count and terminal). We build it explicitly in ``init`` via ``longcat_linspace_sigmas``.
  2. CFG-ZERO (spec blocker #3): guidance is the CFG-zero optimized-scale formula (``st_star`` projection of
     cond onto uncond), NOT ClassicCFG. Implemented as ``CFGZeroPolicy`` (a ``CFGPolicy`` subclass local to
     this recipe). Default ``guidance_scale=1.0`` degenerates the scale term but fastvideo still runs BOTH
     branches whenever ``do_classifier_free_guidance`` is set (independent of the scale), so we keep both.
  3. SIGN CONVENTION (spec blocker #1): handled in the ``LongCatDiT`` adapter (returns ``-velocity``), so the
     loop's ``FLOW_MATCH_STEP`` (``x + (sigma_next-sigma)*v``) reproduces the stage's negate-then-step. The
     CFG-zero combine here operates on the (already-negated) velocities; the combine is sign-equivariant
     (``st_star`` is a ratio of dot-products, and the affine combine is linear), so combining the negated
     velocities equals negating the combine of the raw ones — i.e. ``-combine(raw) == combine(-raw)`` — which
     is exactly fastvideo's order (combine raw, then negate). The toy backend has no negating adapter; that is
     fine for the CPU control-flow check (we still exercise the schedule + CFG-zero + Euler step end-to-end).

Geometry is Wan2.1's (``AutoencoderKLWan``: 16 channels, 4x temporal, 8x spatial). All per-request state lives
in ``LoopState`` (interleave-safe), matching the canonical ``WanDenoiseLoop`` contract.
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
from v2.loop.policies import CFGPolicy
from v2.platform import FLOW_MATCH_STEP
from v2.recipes.longcat.sampler import longcat_linspace_sigmas
from v2.recipes.wan21.loop import latent_shape

# LongCat reuses Wan2.1 VAE geometry (AutoencoderKLWan: z=16, 4x temporal, 8x spatial).
LONGCAT_LATENT_CHANNELS = 16
LONGCAT_TEMPORAL_RATIO = 4
LONGCAT_SPATIAL_RATIO = 8


class CFGZeroPolicy(CFGPolicy):
    """CFG-zero optimized-scale guidance (LongCat; from the CFG-zero* paper) over the standard
    cond/uncond two-forward decomposition.

    Per sample, project cond onto uncond to get the optimized scale
    ``st_star = <v_cond, v_uncond> / (||v_uncond||^2 + 1e-8)``, then combine
    ``out = v_uncond*st_star + scale*(v_cond - v_uncond*st_star)``. This matches
    ``LongCatDenoisingStage.optimized_scale`` + its CFG application (longcat_denoising.py). A v2 step works on
    ONE unbatched sample ([C,T,H,W]), so ``st_star`` is a single scalar over the whole tensor (B=1)."""

    def combine(self, preds, guidance_scale, ctx, state):
        cond = np.asarray(preds["cond"], dtype=np.float64)
        uncond = np.asarray(preds["uncond"], dtype=np.float64)
        # st_star = <cond, uncond> / (||uncond||^2 + 1e-8) over the full (B=1) tensor — fastvideo reshapes to
        # [B,-1] and reduces along dim=1; here the single sample is the whole flattened tensor.
        dot = float(np.sum(cond * uncond))
        sq = float(np.sum(uncond * uncond)) + 1e-8
        st_star = dot / sq
        out = uncond * st_star + float(guidance_scale) * (cond - uncond * st_star)
        return out.astype("float32")


class LongCatDenoiseLoop:
    """Driven flow-match denoise loop for LongCat-Video T2V. Same four-method contract as
    ``WanDenoiseLoop``; the deltas are the explicit linspace sigma schedule and the CFG-zero combine."""

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 precision,
                 expert,
                 cost,
                 sigma_min: float = 0.001,
                 latent_channels: int = LONGCAT_LATENT_CHANNELS,
                 spatial_ratio: int = LONGCAT_SPATIAL_RATIO,
                 temporal_ratio: int = LONGCAT_TEMPORAL_RATIO):
        self.loop_id = loop_id
        self.cfg = cfg  # CFGZeroPolicy
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.sigma_min = sigma_min  # the linspace terminal (LongCat: 0.001, NOT 0.0)
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        # LongCat schedule (spec blocker #2): linspace(1.0, 0.001, num_steps) + a trailing 0.0 sentinel so
        # next() always has a sigma_next for the final boundary. NOT flow-shift, NOT terminal-0 linspace.
        sig = longcat_linspace_sigmas(req.diffusion.num_steps, sigma_min=self.sigma_min)
        shape = latent_shape(req,
                             model,
                             channels=self.latent_channels,
                             spatial_ratio=self.spatial_ratio,
                             temporal_ratio=self.temporal_ratio)
        # Initial noise x = randn * sigma[0] (=1.0). LongCat's LatentPreparationStage multiplies by
        # scheduler.init_noise_sigma (=1.0 for FlowMatchEulerDiscreteScheduler), so this matches; the refine
        # branch (refine_from/stage1_video) is the path that SKIPS this re-scale, but base T2V always scales.
        x = (rng.standard_normal(shape) * float(sig[0])).astype("float32")
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"video": x},
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) * 1000.0 for s in sig])  # model timestep = sigma*1000 (Wan convention)
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
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
        x = st.latents["video"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision
        # LongCat runs BOTH branches whenever a negative prompt is present (do_classifier_free_guidance is
        # independent of the scale value); CFG-zero degenerates only its scale term, not the branch set.
        do_cfg = ne is not None
        branches = ["cond", "uncond"] if do_cfg else ["cond"]

        def _velocity(model: Any, x_: Any, sigma_t_: float, pe_: Any, ne_: Any, scale_: float) -> np.ndarray:
            dit = model.component(expert_id)
            preds = {b: dit(x_, pe_ if b == "cond" else ne_, sigma_t_) for b in branches}
            if not do_cfg:
                return precision.cast(np.asarray(preds["cond"], dtype="float32"))
            return precision.cast(cfg.combine(preds, scale_, sctx, cfg_state))

        def run(model: Any, override: Any = None) -> StepResult:
            kernels = model.platform.kernels
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                velocity = _velocity(model, x, sigma_t, pe, ne, scale)
            x_next = kernels.get(FLOW_MATCH_STEP)(precision.cast(x), velocity, sigma_t, sigma_next)
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": x_next.astype("float32")
            })

        cond_bytes = sum(int(np.asarray(e).nbytes) for e in (pe, ne) if e is not None)
        res = ResourceRequest(compute_seconds=self.cost.predict(int(np.prod(x.shape)), float(len(branches))),
                              resident_bytes=int(x.nbytes) + cond_bytes,
                              peak_activation_bytes=int(x.nbytes))
        return WorkPlan(
            loop_id=self.loop_id,
            instance_id=st.instance_id,
            kind=WorkUnitKind.DIFFUSION_STEP,
            shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                     dims=tuple(x.shape),
                                     dtype=precision.compute_dtype,
                                     extra=(("cfg", type(cfg).__name__), )),
            resources=res,
            payload={
                "branch": "combined",
                "step": i
            },
            run=run,
            label=f"longcat.denoise.{i}",
            # CFG-zero combine is host-side (np.sum / per-sample st_star) -> eager-break under capture, like
            # the Cosmos x0-space CFG. The deterministic Euler step itself is capturable, but the combine is
            # data-dependent on the two branch outputs, so keep the whole step eager for correctness.
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
