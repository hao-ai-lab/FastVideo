"""SD3DenoiseLoop — a thin flow-match Euler denoise loop for Stable Diffusion 3.5 (image / 4D latents).

SD3.5 is a plain FLOW-MATCH MMDiT, so this loop reuses the canonical pieces (``FLOW_MATCH_STEP`` kernel,
``FlowShiftPolicy.build_schedule``, ``ClassicCFG``) — the only deltas vs ``WanDenoiseLoop`` are:

  * IMAGE modality / 4D latents: the latent is ``[16, h//8, w//8]`` (num_frames=1), with NO temporal
    dim. The fastvideo stage carries a fake T dim in its 5D ForwardBatch (unsqueeze(2)/squeeze(2)) but
    the DiT forward itself is strictly 4D — so we drop the T dim here (BRINGUP blocker 3).
  * DUAL DiT conditioning (BRINGUP blocker 1): the SD3 forward needs BOTH the assembled triple-encoder
    joint embed (``encoder_hidden_states``) AND a separate pooled vector (``pooled_projections``). The
    program writes both into slots; this loop threads the joint embed as ``text_embed`` and the pooled
    vector as the positional ``context`` arg — so the dit-call stays the cosmos2-style
    ``dit(latent, text_embed, sigma, context)`` that ToyDiT accepts (CPU-verifiable) and SD3DiT unpacks.
  * VELOCITY output: SD3DiT returns the flow-match velocity directly (no x0 reconstruction), so the
    Euler update is the plain ``x_next = x + (sigma_next - sigma)*v``.

Faithful to ``fastvideo/pipelines/stages/sd35_conditioning.py:SD35DenoisingStage`` (uncond + s·(cond −
uncond) CFG over the FlowMatchEuler schedule). The σ schedule is the flow-shift linspace; ``shift`` comes
from the diffusers ``FlowMatchEulerDiscreteScheduler`` config (3.0 for SD3.5-medium) — see ``card.py``.
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
from v2.platform import FLOW_MATCH_STEP
from v2.platform.backends.toy import LATENT_CHANNELS

# Real SD3.5 AutoencoderKL: z=16, 8x spatial compression, IMAGE (no temporal dim).
SD3_LATENT_CHANNELS = 16
SD3_SPATIAL_RATIO = 8


def sd3_latent_shape(req,
                     model=None,
                     *,
                     channels: int = SD3_LATENT_CHANNELS,
                     spatial_ratio: int = SD3_SPATIAL_RATIO) -> tuple[int, int, int]:
    """4D image latent geometry ``[C, h, w]`` (the leading batch dim is added by the adapter). On the GPU
    backend the true SD3.5 geometry is required; the CPU toy uses a tiny deterministic stand-in (matching
    ``v2.recipes.wan21.loop.latent_shape`` but with NO temporal axis — image, not video)."""
    d = req.diffusion
    if model is not None and getattr(getattr(model, "platform", None), "device", "cpu") == "cuda":
        return (channels, max(1, d.height // spatial_ratio), max(1, d.width // spatial_ratio))
    h = max(2, d.height // 120)
    w = max(2, d.width // 120)
    return (LATENT_CHANNELS, h, w)


class SD3DenoiseLoop:
    """Flow-match Euler denoise for SD3.5 (image / 4D latents, dual text conditioning)."""

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 latent_channels: int = SD3_LATENT_CHANNELS,
                 spatial_ratio: int = SD3_SPATIAL_RATIO):
        self.loop_id = loop_id
        self.cfg = cfg
        self.flow_shift = flow_shift
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        sig = self.flow_shift.build_schedule(req.diffusion.num_steps,
                                             req.diffusion.height,
                                             req.diffusion.width,
                                             sigmas=req.diffusion.sigmas or None)
        shape = sd3_latent_shape(req, model, channels=self.latent_channels, spatial_ratio=self.spatial_ratio)
        x = (rng.standard_normal(shape) * float(sig[0])).astype("float32")
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"image": x},
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) * 1000.0 for s in sig])
        # Dual text conditioning (BRINGUP blocker 1): the program's SD3 text-encode node writes the
        # assembled joint embed AND the pooled vector for both the cond and uncond branches into slots.
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.cond["pooled"] = ctx.slots.get("pooled_projections")
        st.cond["neg_pooled"] = ctx.slots.get("neg_pooled_projections")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_image"] = bool(req.outputs.stream.get("image") or req.outputs.stream.get("video"))
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
        x = st.latents["image"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        pp, npp = st.cond["pooled"], st.cond["neg_pooled"]
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision

        def _velocity(model: Any) -> np.ndarray:
            # The conditioned forward + CFG combine. Each branch threads its OWN pooled vector via the
            # positional ``context`` arg (so SD3DiT carries pooled_projections alongside the joint embed,
            # and ToyDiT harmlessly mean-pools it). Solver/forward dispatch through the kernel table.
            dit = model.component(expert_id)
            pooled_for = {"cond": pp, "uncond": npp}
            embed_for = {"cond": pe, "uncond": ne}
            preds = {b: dit(x, embed_for[b], sigma_t, context=pooled_for[b]) for b in branches}
            return precision.cast(cfg.combine(preds, scale, sctx, cfg_state))

        def run(model: Any, override: Any = None) -> StepResult:
            kernels = model.platform.kernels
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                velocity = _velocity(model)
            x_next = kernels.get(FLOW_MATCH_STEP)(precision.cast(x), velocity, sigma_t, sigma_next)
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": x_next.astype("float32")
            })

        cond_bytes = sum(int(np.asarray(e).nbytes) for e in (pe, ne, pp, npp) if e is not None)
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
                                     extra=(("cfg", type(cfg).__name__), ("image", True))),
            resources=res,
            payload={
                "branch": "combined",
                "step": i
            },
            run=run,
            label=f"sd35.denoise.{i}",
            # Dual-conditioning forward with a per-branch pooled vector threaded outside the workspace ->
            # keep the eager path (no static-buffer graph capture wired for this recipe). The deterministic
            # ODE step itself is capturable, but wiring the full capture form is a later GPU-side concern.
            capturable=False)

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["image"] = result.output["latents"]
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({
                "step": st.step_idx,
                "sigma": st.sigmas[st.step_idx],
                "velocity": np.asarray(result.output["noise_pred"]).copy(),
                "latents": np.asarray(st.latents["image"]).copy(),
            })
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["image"]},
                          metrics={"denoise_steps": float(st.step_idx)},
                          behavior=st.trajectory or None)
