"""LingBotWorldDenoiseLoop — Wan flow-match denoise + boundary-routed MoE, extended with (a) the camera
(Plucker) conditioning slot threaded to the DiT and (b) DUAL classifier-free guidance.

LingBot-World-Base-Cam is Wan2.2-I2V-A14B i2v conditioning (CLIP + first-frame ``[mask|cond]`` latent)
plus a camera/Plucker tensor. The denoise math is otherwise the canonical Wan flow-match Euler step with
``BoundaryTimestepRouting`` between the high-noise ``transformer`` and the low-noise ``transformer_2``. Two
deltas vs ``WanDenoiseLoop`` (faithful to ``LingBotWorldImageToVideoPipeline`` == the Wan i2v denoise
stage, denoising.py:352-375):

  * DUAL guidance — fastvideo switches ``guidance_scale -> guidance_scale_2`` at the SAME expert boundary:
    ``t >= boundary_timestep`` (the high-noise expert) uses ``guidance_scale``; below it (the low-noise
    expert) uses ``guidance_scale_2``. The boundary is in timestep space ``boundary_ratio*1000``, i.e.
    sigma-space ``ctx.sigma >= boundary`` (preset ``boundary_ratio=0.947``). The preset sets both scales
    to 5.0, so the registered path is numerically a single scale — but the loop honors a distinct
    ``guidance_scale_2`` whenever the request carries one.
  * CAMERA — the program's camera node writes ``c2ws_plucker_emb`` into slots; the loop publishes it onto
    the active DiT adapter (``dit.c2ws_plucker_emb = ...``) before the forward, because the v2 dit-call
    surface (and ``ToyDiT``) has no such kwarg. On the CPU toy backend the slot is ``None`` -> the
    degenerate no-camera path (a plain Wan i2v step), so the loop CPU-verifies end-to-end.

BRINGUP (needs request-API extension): the Plucker tensor is built from a per-request camera trajectory
(``poses.npy``/``intrinsics.npy``); v2's ``Request`` has no camera slot yet, so the program node writes
``None`` unless an ``action_path`` override is supplied. GPU-verify (a) the 2x14B MoE CPU-offload swap,
(b) the camera FiLM injection actually steers the generation, (c) the dual-scale boundary parity.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2.loop.contracts import Done, LoopState, StepContext, StepResult, WorkPlan
from v2.recipes.wan21.loop import WanDenoiseLoop

# Wan2.2 boundary expressed in (shifted) sigma space; LingBot-World uses 0.947 (much higher than Wan2.2's
# 0.875/0.9), so the low-noise expert ``transformer_2`` runs almost the whole trajectory.
LINGBOTWORLD_BOUNDARY = 0.947


class LingBotWorldDenoiseLoop(WanDenoiseLoop):
    """Wan i2v denoise + camera Plucker conditioning + dual-guidance boundary switch."""

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 boundary=LINGBOTWORLD_BOUNDARY,
                 latent_channels=16,
                 spatial_ratio=8,
                 temporal_ratio=4):
        super().__init__(loop_id=loop_id,
                         cfg=cfg,
                         flow_shift=flow_shift,
                         precision=precision,
                         expert=expert,
                         cost=cost,
                         latent_channels=latent_channels,
                         spatial_ratio=spatial_ratio,
                         temporal_ratio=temporal_ratio)
        # The boundary the dual-guidance switch keys on; default to the routing policy's own boundary so the
        # CFG switch and the expert switch fire on the SAME step (matching the fastvideo denoise stage).
        self.boundary = float(getattr(expert, "boundary", boundary))

    def init(self, req, model, ctx) -> LoopState:
        st = super().init(req, model, ctx)
        # Dual guidance: the second CFG scale used below the boundary (low-noise expert). Fall back to the
        # primary scale when the request doesn't carry a distinct guidance_scale_2 (preset sets both 5.0).
        gs2 = getattr(req.diffusion, "guidance_scale_2", None)
        st.scratch["guidance_scale_2"] = float(gs2) if gs2 is not None else float(req.diffusion.guidance_scale)
        # Camera / Plucker conditioning written by the program's camera node (None == no-camera path).
        st.scratch["c2ws_plucker_emb"] = ctx.slots.get("c2ws_plucker_emb")
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
        # DUAL guidance: high-noise (ctx.sigma>=boundary) -> guidance_scale; low-noise -> guidance_scale_2.
        scale = (st.scratch["guidance_scale"] if sigma_t >= self.boundary else st.scratch["guidance_scale_2"])
        cfg, precision = self.cfg, self.precision
        i2v_ctx, i2v_cond = st.scratch.get("i2v_img_embeds"), st.scratch.get("i2v_cond")
        plucker = st.scratch.get("c2ws_plucker_emb")

        def _velocity(model: Any, x_: np.ndarray, sigma_t_: float, pe_: Any, ne_: Any, scale_: float) -> np.ndarray:
            # The conditioned forward + CFG combine. The active expert's adapter gets the per-step camera
            # tensor published out-of-band (the v2 dit-call surface has no c2ws_plucker_emb kwarg); ToyDiT
            # simply ignores the attribute, so the CPU path is the no-camera step.
            dit = model.component(expert_id)
            if plucker is not None and hasattr(dit, "c2ws_plucker_emb"):
                dit.c2ws_plucker_emb = plucker
            preds = {
                b: dit(x_, pe_ if b == "cond" else ne_, sigma_t_, context=i2v_ctx, cond=i2v_cond)
                for b in branches
            }
            return precision.cast(cfg.combine(preds, scale_, sctx, cfg_state))

        def run(model: Any, override: dict | None = None) -> StepResult:
            from v2.platform import FLOW_MATCH_STEP
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

        from v2._enums import WorkUnitKind
        from v2.loop.contracts import ResourceRequest, ShapeSignature
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
                                     extra=(("cfg", type(cfg).__name__), ("camera", plucker is not None))),
            resources=res,
            payload={
                "branch": "combined",
                "step": i
            },
            run=run,
            label=f"lingbotworld.denoise.{i}",
            # Camera (out-of-band adapter state) + dual-scale boundary thread non-workspace conditioning,
            # exactly like Wan i2v -> eager path (no CUDA-graph capture).
            capturable=False)
