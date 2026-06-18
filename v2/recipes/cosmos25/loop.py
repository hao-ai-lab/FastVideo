"""Cosmos25DenoiseLoop — a flow-match Euler denoiser with Cosmos2.5's per-frame sigma timestep.

Cosmos-Predict2.5 IS a rectified-flow / flow-match model (unlike Cosmos-Predict2, which is an EDM
denoiser). The sigma schedule and the deterministic Euler solver are EXACTLY Wan's
(``FlowShiftPolicy.build_schedule`` + ``FLOW_MATCH_STEP``), and CFG is the standard ``ClassicCFG``
(``uncond + s*(cond-uncond)``). So this loop is almost identical to ``WanDenoiseLoop``; it forks it for
ONE load-bearing reason (blocker (1)):

  * Wan feeds the DiT a SCALAR ``timestep = sigma * num_train_timesteps`` (``sigma*1000``).
  * Cosmos2.5 feeds a PER-FRAME 2D ``timestep[B,T]`` of the PLAIN sigma in [0,1] — the fastvideo stage
    computes ``t * 0.001`` of the 0..1000 schedule, which is precisely the shifted sigma. So we set
    ``st.timesteps = st.sigmas`` (NO *1000) and the ``Cosmos25DiT`` adapter broadcasts that plain sigma
    to ``[B, T]`` and builds the mandatory zero ``condition_mask`` / ones ``padding_mask`` + ``fps`` the
    forward requires (the model concats the masks itself — we feed the RAW 16ch latent). See
    ``v2/platform/backends/torch_cosmos25.py:Cosmos25DiT`` and the faithful source
    ``fastvideo/pipelines/stages/denoising.py:Cosmos25DenoisingStage``.

Verification that the math matches the source: the fastvideo ``FlowMatchEulerDiscreteScheduler``
(shift=5.0) builds ``sigma' = shift*sigma/(1+(shift-1)*sigma)`` over a 1->0 base + a terminal 0, sets
``timesteps = sigma'*1000``, feeds the DiT ``timesteps*0.001 = sigma'``, and steps
``prev = sample + (sigma_next - sigma)*velocity`` — i.e. exactly ``build_flow_sigmas(num_steps,
shift=5.0)`` + ``FLOW_MATCH_STEP``.

video2world / image2world (frame_replace) conditioning (cond_indicator GT-x0 clamp + GT-velocity
injection, conditional_frame_timestep=0.1 on conditioned frames) is present in fastvideo
(``Cosmos25V2WDenoisingStage``) but is NOT in the registered HF presets — this loop scopes to t2v.
The hooks are read from ``ctx.slots`` (None for t2v -> inert), mirroring the Wan/Cosmos2 pattern, so a
later capability adds the gated injection without changing the t2v path. (BRINGUP: frame-replace needs a
request-API extension to carry the conditioning latents + indicator.)
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
from v2.recipes.wan21.loop import latent_shape

# Real Cosmos2.5 Wan-style VAE (Cosmos25WanVAE) compression: z_dim=16, 4x temporal, 8x spatial — the
# same geometry as Wan2.1, so ``latent_shape`` (reused from the Wan recipe) yields
# (16, (F-1)//4 + 1, H//8, W//8) on the real backend.
COSMOS25_LATENT_CHANNELS = 16
COSMOS25_TEMPORAL_RATIO = 4
COSMOS25_SPATIAL_RATIO = 8
COSMOS25_DEFAULT_FPS = 24


class Cosmos25DenoiseLoop:
    """Flow-match Euler denoise loop with Cosmos2.5's plain per-frame sigma timestep convention."""

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 latent_channels: int = COSMOS25_LATENT_CHANNELS,
                 spatial_ratio: int = COSMOS25_SPATIAL_RATIO,
                 temporal_ratio: int = COSMOS25_TEMPORAL_RATIO):
        self.loop_id = loop_id
        self.cfg = cfg
        self.flow_shift = flow_shift
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        # The σ schedule + flow-shift are Wan's verbatim (shift=5.0 for Cosmos2.5): 1 -> 0 over
        # num_steps+1 points with the flow-shift warp applied.
        sig = self.flow_shift.build_schedule(req.diffusion.num_steps,
                                             req.diffusion.height,
                                             req.diffusion.width,
                                             sigmas=req.diffusion.sigmas or None)
        shape = latent_shape(req,
                             model,
                             channels=self.latent_channels,
                             spatial_ratio=self.spatial_ratio,
                             temporal_ratio=self.temporal_ratio)
        # Flow-match noise init: x = randn * sigma_0 (sigma_0 == 1 for an unshifted-terminal schedule).
        x = (rng.standard_normal(shape) * float(sig[0])).astype("float32")
        st = LoopState(
            loop_id=self.loop_id,
            instance_id=model.card.model_id,
            request_id=req.request_id,
            profile=ctx.profile,
            rng=rng,
            seed=seed,
            latents={"video": x},
            sigmas=[float(s) for s in sig],
            # BLOCKER (1): the model timestep is the PLAIN sigma (NOT sigma*1000). The
            # Cosmos25DiT adapter broadcasts it to the per-frame [B, T] form the forward needs.
            timesteps=[float(s) for s in sig])
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        st.scratch["fps"] = int(getattr(req.diffusion, "fps", None) or COSMOS25_DEFAULT_FPS)
        # video2world (frame_replace) conditioning: None for the t2v preset -> the gated injection a later
        # capability would add is inert (pure t2v, matching the fastvideo pipeline with no image/video).
        st.scratch["conditioning_latents"] = ctx.slots.get("conditioning_latents")
        st.scratch["cond_indicator"] = ctx.slots.get("cond_indicator")
        st.plugin_state["cfg"] = {}
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        # The model timestep here is the PLAIN sigma (st.timesteps[i] == st.sigmas[i]); the adapter turns
        # it into the per-frame [B, T] timestep + masks + fps internally.
        expert_id = self.expert.expert_for(StepContext(i, st.timesteps[i], sigma_t))
        sctx = StepContext(step_idx=i, timestep=st.timesteps[i], sigma=sigma_t, active_expert_id=expert_id)
        cfg_state = st.plugin_state["cfg"]
        branches = self.cfg.branches_this_step(sctx, cfg_state)
        x = st.latents["video"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision

        def _velocity(model: Any, x_: Any, sigma_t_: float, pe_: Any, ne_: Any, scale_: float) -> np.ndarray:
            # The conditioned forward + ClassicCFG combine. The dit-call is toy-compatible:
            # ``dit(latent, text_embed, sigma)`` — the Cosmos2.5-specific timestep/masks/fps are built
            # INSIDE the Cosmos25DiT adapter (so the CPU ToyDiT, which ignores them, also runs).
            dit = model.component(expert_id)
            preds = {b: dit(x_, pe_ if b == "cond" else ne_, sigma_t_) for b in branches}
            return precision.cast(cfg.combine(preds, scale_, sctx, cfg_state))

        def run(model, override=None):
            # The EAGER thunk: handles the override (cached-prediction) path. The deterministic Euler
            # solver dispatches through the platform kernel table (numpy on CPU, the device kernel on GPU).
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

        def graph_fn(model, ws):
            # The CAPTURABLE op-structure: reads EVERY per-step input from the static workspace and writes
            # into the static output buffer, so the captured graph replays correctly with rebound buffers.
            st_t, st_n = float(ws["sigma_t"]), float(ws["sigma_next"])
            velocity = _velocity(model, ws["x"], st_t, ws["pe"], ws["ne"], float(ws["scale"]))
            x_next = model.platform.kernels.get(FLOW_MATCH_STEP)(precision.cast(ws["x"]), velocity, st_t, st_n)
            np.copyto(ws["out"], x_next.astype("float32"))
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": np.array(ws["out"], copy=True)
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
            label=f"cosmos25.denoise.{i}",
            # The flow-match Euler step is a clean capturable op (no host RNG / data-dependent branch),
            # like Wan t2v: the branch-set + expert + scheduler-precision discriminate the captured graph.
            capturable=True,
            graph_key=(tuple(sorted(branches)), expert_id, precision.scheduler_step_in_fp32),
            graph_fn=graph_fn,
            graph_inputs={
                "x": x,
                "sigma_t": sigma_t,
                "sigma_next": sigma_next,
                "pe": pe,
                "ne": ne,
                "scale": scale
            })

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
