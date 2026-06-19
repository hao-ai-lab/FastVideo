"""GameCraftDenoiseLoop — flow-match Euler denoise with GameCraft i2v + camera conditioning.

GameCraft (HunyuanGameCraft) is an interactive world model: a camera/action-conditioned i2v diffusion DiT.
The sampler is plain flow-match (``FlowMatchEulerDiscreteScheduler``, ``flow_shift=5.0``,
``timestep = σ·1000``) with standard ClassicCFG (``guidance_scale=6.0``, ``guidance=None`` to the DiT, no
embedded guidance), so the integrator is identical to ``WanDenoiseLoop`` (``x_next = x + (σ_next−σ)·v`` via
``FLOW_MATCH_STEP``). The deltas this loop carries (faithful to
``fastvideo/pipelines/stages/gamecraft_denoising.py``):

  1. **33-channel input is assembled by the DiT adapter, not here.** The DiT forward wants
     ``cat([latent16 | gt_latent16 | mask1], dim=1)``; the loop hands the bare 16ch latent + a ``cond``
     pack ``{"gt_latents","mask","camera_states"}`` to the adapter, which concats. The toy ``ToyDiT``
     accepts and ignores the ``cond`` kwarg, so this loop CPU-verifies unchanged.

  2. **Per-step clean-ref injection (frame_replace).** Official GameCraft overwrites the conditioned latent
     frames (``mask>0.5``) with the clean reference latent at every step (not just step 0) to keep them
     noise-free. ``next()`` mutates the working latent before the forward when a reference latent is present.

  3. **Dual text encoders.** ``encoder_hidden_states = [LLaMA_states(4096d), CLIP_pooled(768d)]``. The loop
     threads the LLaMA states as the usual ``text_embed`` (cond vs uncond) and the CLIP pooled embed
     through the adapter's ``context=`` slot per branch, so ClassicCFG combines both correctly.

  4. **BRINGUP — camera/action conditioning.** GameCraft is CameraNet (Plücker) conditioned, with the
     forward special-casing the latent temporal length (18/9/10) for autoregressive multi-chunk generation.
     v2 has no camera/action request field or Plücker-conversion node yet, so the registered path passes
     ``camera_states=None`` -> the DiT's ``if camera_states is not None`` branch is skipped (the degenerate
     denoise). For pure t2v (no reference image) gt_latents+mask are zero -> the 33ch concat reduces to
     ``[noise | 0 | 0]``, exactly the ``GameCraftDenoisingStage`` fallback. Wiring real camera/action + the
     18/9/10 length special-casing needs a request-API extension (see ``program.py`` blockers).

All per-request state lives in ``LoopState`` (interleave-safe). ``next`` is kernel-free; ``advance`` folds
the result and, under ROLLOUT, captures a trajectory slice — same shape as ``WanDenoiseLoop``.
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

# GameCraftVAE geometry: z=16, 8x spatial, 4x temporal (T_lat = (num_frames-1)//4 + 1).
GAMECRAFT_LATENT_CHANNELS = 16
GAMECRAFT_TEMPORAL_RATIO = 4
GAMECRAFT_SPATIAL_RATIO = 8


class GameCraftDenoiseLoop:
    """Flow-match Euler denoise with GameCraft i2v (33ch concat + per-step ref injection) + camera hooks."""

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 latent_channels: int = GAMECRAFT_LATENT_CHANNELS,
                 spatial_ratio: int = GAMECRAFT_SPATIAL_RATIO,
                 temporal_ratio: int = GAMECRAFT_TEMPORAL_RATIO):
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
        # Flow-match σ schedule (shift=5.0 for GameCraft); honor an explicit distilled sigmas list if given.
        sig = self.flow_shift.build_schedule(req.diffusion.num_steps,
                                             req.diffusion.height,
                                             req.diffusion.width,
                                             sigmas=req.diffusion.sigmas or None)
        shape = latent_shape(req,
                             model,
                             channels=self.latent_channels,
                             spatial_ratio=self.spatial_ratio,
                             temporal_ratio=self.temporal_ratio)
        x = (rng.standard_normal(shape) * float(sig[0])).astype("float32")
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"video": x},
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) * 1000.0 for s in sig])  # model timestep = σ·1000
        # Dual text: LLaMA hidden states (cond/uncond) ride the usual prompt-embed slots; the CLIP pooled
        # embeds (cond/uncond) ride dedicated slots and are threaded through the adapter's ``context=``.
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["clip_pos"] = ctx.slots.get("clip_text_embeds")
        st.scratch["clip_neg"] = ctx.slots.get("clip_neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        # i2v conditioning slots written by the image-cond program node. All None for pure t2v -> the 33ch
        # concat degenerates to [noise | 0 | 0] and the ref-injection branch is inert (the
        # GameCraftDenoisingStage fallback). camera_states is a BRINGUP slot (no request plumbing yet).
        st.scratch["gt_latents"] = ctx.slots.get("gt_latents")
        st.scratch["conditioning_mask"] = ctx.slots.get("conditioning_mask")
        st.scratch["ref_latent_for_injection"] = ctx.slots.get("ref_latent_for_injection")
        st.scratch["camera_states"] = ctx.slots.get("camera_states")  # BRINGUP: None until camera plumbing
        st.plugin_state["cfg"] = {}
        return st

    def _inject_ref(self, x: np.ndarray, mask, ref) -> np.ndarray:
        """Per-step frame_replace: overwrite conditioned frames (mask>0.5) with the clean reference latent.

        Faithful to ``GameCraftDenoisingStage`` (``latents[:,:,t_idx] = ref[:,:,0]`` for each conditioned
        ``t``). Loop latents here are unbatched ``[C,T,h,w]`` (toy/CPU geometry), so the mask's temporal axis
        is read at ``mask[0,0,:,0,0]`` when 5D (the fastvideo ``[B,1,T,H,W]`` mask) or as a 1D ``[T]`` vector
        otherwise. ``ref`` is the clean ref latent ``[C,1,h,w]`` (or ``[C,T,h,w]``)."""
        m = np.asarray(mask)
        if m.ndim == 5:  # fastvideo-shaped [B,1,T,H,W]: read the temporal slice
            frame_mask = m[0, 0, :, 0, 0]
        elif m.ndim == 4:  # loop-shaped [1,T,h,w]: collapse the spatial axes
            frame_mask = m[0, :, 0, 0]
        elif m.ndim == 1:  # already a per-frame vector [T]
            frame_mask = m
        else:
            return x
        r = np.asarray(ref, dtype="float32")
        ref_frame = r[:, 0] if r.ndim >= 4 else r  # [C,h,w]
        out = np.array(x, dtype="float32", copy=True)
        # Frame replacement needs the ref frame to match the working latent's per-frame geometry (always
        # true on the real VAE). If a backend's encode produced a mismatched shape (e.g. the toy VAE that
        # does not spatially downsample), skip injection rather than crash — the degenerate denoise proceeds.
        if ref_frame.shape != out[:, 0].shape:
            return out
        n_t = min(out.shape[1], int(frame_mask.shape[0]))
        for t_idx in range(n_t):
            if float(frame_mask[t_idx]) > 0.5:
                out[:, t_idx] = ref_frame
        return out

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
        gt = st.scratch.get("gt_latents")
        mask = st.scratch.get("conditioning_mask")
        ref = st.scratch.get("ref_latent_for_injection")
        camera = st.scratch.get("camera_states")  # BRINGUP: None for the registered path
        # Per-step clean-ref injection BEFORE the forward (frame_replace) — only when an i2v ref is present.
        if ref is not None and mask is not None:
            x = self._inject_ref(x, mask, ref)
            st.latents["video"] = x

        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        clip_pos, clip_neg = st.scratch.get("clip_pos"), st.scratch.get("clip_neg")
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision
        # One ``cond`` pack reused across CFG branches (gt/mask/camera are branch-independent); the adapter
        # assembles the 33ch concat from it. None entries -> zero gt/mask and no camera inside the adapter.
        cond_pack = {"gt_latents": gt, "mask": mask, "camera_states": camera}

        def _velocity(model: Any, x_: np.ndarray, sigma_t_: float, pe_: Any, clip_: Any, ne_: Any, clip_n_: Any,
                      scale_: float) -> np.ndarray:
            # Conditioned forward + ClassicCFG combine. The CLIP pooled embed rides ``context=`` per branch
            # so text_states_2 differs cond vs uncond; ``cond`` carries the i2v/camera pack. The toy DiT
            # accepts and ignores both kwargs, so this is CPU-toy compatible.
            dit = model.component(expert_id)
            preds = {
                b:
                dit(x_,
                    pe_ if b == "cond" else ne_,
                    sigma_t_,
                    context=(clip_ if b == "cond" else clip_n_),
                    cond=cond_pack)
                for b in branches
            }
            return precision.cast(cfg.combine(preds, scale_, sctx, cfg_state))

        def run(model, override=None):
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                velocity = _velocity(model, x, sigma_t, pe, clip_pos, ne, clip_neg, scale)
            x_next = model.platform.kernels.get(FLOW_MATCH_STEP)(precision.cast(x), velocity, sigma_t, sigma_next)
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": x_next.astype("float32")
            })

        cond_bytes = sum(int(np.asarray(e).nbytes) for e in (pe, ne, clip_pos, clip_neg, gt, mask) if e is not None)
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
                                     extra=(("cfg", type(cfg).__name__), ("gamecraft", True))),
            resources=res,
            payload={
                "branch": "combined",
                "step": i
            },
            run=run,
            label=f"gamecraft.denoise.{i}",
            # i2v ref-injection mutates host state and threads non-workspace conditioning each step, which
            # static-buffer capture would miss, so this step takes the eager path (like cosmos2's gated
            # frame-injection). Pure-t2v could capture, but keeping one path is simpler and correct.
            capturable=False)

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["video"] = result.output["latents"]
        if st.profile == ExecutionProfile.ROLLOUT:
            i = st.step_idx
            st.trajectory.append({
                "step": i,
                "sigma": st.sigmas[i],
                "velocity": np.asarray(result.output["noise_pred"]).copy(),
                "latents": np.asarray(st.latents["video"]).copy()
            })
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["video"]},
                          metrics={"denoise_steps": float(st.step_idx)},
                          behavior=st.trajectory or None)
