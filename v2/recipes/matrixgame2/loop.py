"""MatrixGame2CausalDMDLoop — few-step, causal/autoregressive, action-conditioned DMD denoise loop for
Matrix-Game 2.0. Cannot reuse ``WanDenoiseLoop``/``CosmosDenoiseLoop`` because:

  (a) Few-step DMD, NOT a flow-match velocity ODE. The model output is EPSILON; each DMD step converts
      ``eps -> x0`` via ``pred_video = x - sigma_t*eps`` (scheduler sigma table), then RE-ADDS noise to
      the next DMD timestep. Fixed 3-step schedule ``[1000,666,333]``, warped through the FlowUniPC
      timestep grid (``warp_denoising_step``). CFG is OFF (guidance_scale=1.0).
  (b) Causal block-autoregressive: the video latent is split into blocks of ``num_frames_per_block`` (=3);
      each block is fully DMD-denoised, then a context pass at ``timestep=context_noise(=0)`` writes its
      CLEAN K/V into a sliding-window KV cache before the next block. Per-frame timestep ``[B, num_frames]``.
  (c) Action-conditioned: per-frame mouse[F,2] / keyboard[F,4] feed ActionModule blocks with their own KV
      caches. The no-action degenerate path is what CPU-verifies here; live action routing is BRINGUP
      (needs a request-API extension — the loop exposes ``mouse_cond``/``keyboard_cond`` slots).

Faithful port of ``fastvideo/pipelines/stages/matrixgame2_denoising.py:MatrixGame2CausalDenoisingStage``.
The DiT call is dispatched two ways so the SAME loop runs on the CPU toy AND the GPU adapter:
  * GPU: ``model.component('transformer')`` is a ``MatrixGame2CausalDiT`` exposing ``.call(...)`` with the
    causal kv_cache / i2v / action plumbing built INTERNALLY (the loop just passes numpy + the cache lists).
  * CPU toy: ``ToyDiT`` has only ``dit(latent, text_embed, sigma)``; ``_eps`` falls back to that (no
    kv_cache — a degenerate stand-in exercising the DMD + block + epsilon->x0 control flow with real
    numbers). The KV caches are still allocated + threaded so the structure is identical.

The GPU kv_cache / crossattn_cache / action-cache shapes come from the loaded arch_config and are
allocated by the GPU adapter's companion init; on CPU we allocate placeholder dict lists the toy ignores.
The MoE high/low-noise boundary path (``pred_noise_to_x_bound`` + ``add_noise_high``) exists for
GTA/TempleRun variants (``boundary_ratio`` set); the Base distilled checkpoint is single-expert.
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
from v2.recipes.matrixgame2.sampler import (
    MATRIXGAME2_CONTEXT_NOISE,
    MATRIXGAME2_DMD_STEPS,
    MATRIXGAME2_NUM_FRAMES_PER_BLOCK,
    add_noise,
    build_flow_unipc_table,
    pred_noise_to_pred_video,
    pred_noise_to_x_bound,
    warp_dmd_timesteps,
)
from v2.recipes.wan21.loop import latent_shape

# Wan2.1 VAE (AutoencoderKLWan): z=16, 4x temporal, 8x spatial. The DiT patch_embedding is 32-in
# (16 noise + 16 i2v cond_concat) but the LOOP's latent is the 16ch noise; the adapter does the concat.
MATRIXGAME2_LATENT_CHANNELS = 16
MATRIXGAME2_TEMPORAL_RATIO = 4
MATRIXGAME2_SPATIAL_RATIO = 8


class MatrixGame2CausalDMDLoop:

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 dmd_steps: tuple[int, ...] = MATRIXGAME2_DMD_STEPS,
                 context_noise: int = MATRIXGAME2_CONTEXT_NOISE,
                 num_frames_per_block: int = MATRIXGAME2_NUM_FRAMES_PER_BLOCK,
                 boundary_ratio: float | None = None,
                 latent_channels: int = MATRIXGAME2_LATENT_CHANNELS,
                 spatial_ratio: int = MATRIXGAME2_SPATIAL_RATIO,
                 temporal_ratio: int = MATRIXGAME2_TEMPORAL_RATIO):
        self.loop_id = loop_id
        self.cfg = cfg  # carried for the WorkPlan op-structure key; CFG is OFF (scale=1.0) for the distilled model
        self.flow_shift = flow_shift
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.dmd_steps = tuple(dmd_steps)
        self.context_noise = int(context_noise)
        self.num_frames_per_block = int(num_frames_per_block)
        self.boundary_ratio = boundary_ratio
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio

    # --------------------------------------------------------------------- #
    # init                                                                   #
    # --------------------------------------------------------------------- #
    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        shift = float(self.flow_shift.shift_for(req.diffusion.height, req.diffusion.width))
        sigmas, sched_timesteps = build_flow_unipc_table(shift=shift)
        # Warp the nominal DMD steps through the FlowUniPC timestep grid (warp_denoising_step=True).
        warped = warp_dmd_timesteps(self.dmd_steps, scheduler_timesteps=sched_timesteps, shift=shift)
        shape = latent_shape(req,
                             model,
                             channels=self.latent_channels,
                             spatial_ratio=self.spatial_ratio,
                             temporal_ratio=self.temporal_ratio)
        # DMD starts from pure noise at the first (largest) warped timestep (sigma ~ 1).
        x = rng.standard_normal(shape).astype("float32")
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"video": x},
                       sigmas=[float(s) for s in sigmas],
                       timesteps=[float(t) for t in warped])
        st.scratch["sched_sigmas"] = sigmas
        st.scratch["sched_timesteps"] = sched_timesteps
        st.scratch["dmd_timesteps"] = [float(t) for t in warped]
        # i2v + interactive conditioning (program writes these into slots; degenerate path -> None).
        st.scratch["image_embeds"] = ctx.slots.get("i2v_img_embeds")  # CLIP 257x1280, sole cross-attn ctx
        st.scratch["image_latent"] = ctx.slots.get("i2v_cond")  # raw cond_latent for channel-concat
        st.scratch["mouse_cond"] = ctx.slots.get("mouse_cond")  # BRINGUP: action surface (None here)
        st.scratch["keyboard_cond"] = ctx.slots.get("keyboard_cond")
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        # Causal block schedule over the latent-frame axis (axis 1 of [C, T, h, w]).
        nf = int(shape[1])
        nfpb = max(1, self.num_frames_per_block)
        # If the latent frame count isn't a multiple, the last block is shorter (the GPU stage asserts
        # divisibility; the CPU toy stays robust so the loop still verifies on any tiny toy shape).
        blocks: list[tuple[int, int]] = []
        start = 0
        while start < nf:
            cur = min(nfpb, nf - start)
            blocks.append((start, cur))
            start += cur
        st.scratch["blocks"] = blocks
        st.scratch["block_idx"] = 0
        st.scratch["dmd_idx"] = 0
        st.scratch["phase"] = "denoise"  # "denoise" (3 DMD steps/block) then "context" (1 clean-K/V pass)
        # Sliding-window KV caches (one dict per transformer block). On CPU these are inert placeholders the
        # toy ignores; on GPU the MatrixGame2CausalDiT adapter mutates them in place across blocks/steps.
        st.scratch["kv_cache"] = []
        st.scratch["crossattn_cache"] = []
        st.scratch["kv_cache_mouse"] = []
        st.scratch["kv_cache_keyboard"] = []
        st.plugin_state["cfg"] = {}
        # On GPU the MatrixGame2CausalDiT adapter OWNS the sliding-window KV / cross-attn caches (their
        # shapes come from the loaded arch); reset them once per request so a fresh generate starts clean.
        try:
            dit = model.component("transformer")
            if hasattr(dit, "reset_caches"):
                dit.reset_caches()
        except Exception:
            pass  # CPU toy has no caches to reset
        return st

    # --------------------------------------------------------------------- #
    # eps dispatch (GPU causal adapter vs CPU toy)                           #
    # --------------------------------------------------------------------- #
    def _eps(self, dit: Any, latent: np.ndarray, st: LoopState, start_frame: int, num_frames: int, timestep: float,
             frame_seq_len: int) -> np.ndarray:
        """One forward returning the RAW epsilon prediction. Per-frame timestep [num_frames] (long-ish);
        the GPU adapter keeps it 2-D. ``hasattr(dit, 'call')`` selects the causal adapter; else the toy."""
        per_frame_t: np.ndarray = np.full((num_frames, ), float(timestep), dtype="float32")
        if hasattr(dit, "call"):  # MatrixGame2CausalDiT (GPU): full causal/i2v/action plumbing
            return np.asarray(dit.call(
                latent,
                st.scratch.get("image_embeds"),
                per_frame_t,
                kv_cache=st.scratch["kv_cache"] or None,
                crossattn_cache=st.scratch["crossattn_cache"] or None,
                kv_cache_mouse=st.scratch["kv_cache_mouse"] or None,
                kv_cache_keyboard=st.scratch["kv_cache_keyboard"] or None,
                current_start=start_frame * frame_seq_len,
                start_frame=start_frame,
                num_frame_per_block=num_frames,
                mouse_cond=st.scratch.get("mouse_cond"),
                keyboard_cond=st.scratch.get("keyboard_cond"),
                image_latent=st.scratch.get("image_latent"),
            ),
                              dtype="float32")
        # CPU toy degenerate path: dit(latent, text_embed, sigma). Feed the sigma at this timestep so the
        # toy velocity is a deterministic function of the DMD step (exercises the eps->x0 control flow).
        sigma = self._sigma_at(st, timestep)
        ctx_embed = st.scratch.get("image_embeds")
        return np.asarray(dit(latent, ctx_embed, sigma), dtype="float32")

    @staticmethod
    def _sigma_at(st: LoopState, timestep: float) -> float:
        sigmas, tsteps = st.scratch["sched_sigmas"], st.scratch["sched_timesteps"]
        idx = int(np.argmin(np.abs(np.asarray(tsteps, dtype=np.float64) - float(timestep))))
        return float(sigmas[idx])

    def _frame_seq_len(self, x: np.ndarray) -> int:
        # frame_seq_len = (h*w) / (patch_h*patch_w). Wan patch is [1,2,2] -> /4. The toy shape is tiny; clamp.
        h, w = int(x.shape[2]), int(x.shape[3])
        return max(1, (h * w) // 4)

    # --------------------------------------------------------------------- #
    # next                                                                   #
    # --------------------------------------------------------------------- #
    def next(self, st: LoopState):
        blocks = st.scratch["blocks"]
        b_idx = st.scratch["block_idx"]
        if b_idx >= len(blocks):
            return Done()
        start_frame, num_frames = blocks[b_idx]
        phase = st.scratch["phase"]
        dmd_idx = st.scratch["dmd_idx"]
        dmd_timesteps = st.scratch["dmd_timesteps"]
        x_full = st.latents["video"]
        block = x_full[:, start_frame:start_frame + num_frames]  # [C, nf, h, w]
        sigmas, tsteps = st.scratch["sched_sigmas"], st.scratch["sched_timesteps"]
        precision, rng = self.precision, st.rng
        frame_seq_len = self._frame_seq_len(x_full)
        boundary_ts = (self.boundary_ratio * 1000.0) if self.boundary_ratio is not None else None
        _t = dmd_timesteps[dmd_idx] if phase == "denoise" else float(self.context_noise)
        _s = self._sigma_at(st, dmd_timesteps[dmd_idx]) if phase == "denoise" else 0.0
        expert_id = self.expert.expert_for(StepContext(dmd_idx, _t, _s))

        is_context = phase == "context"
        timestep = float(self.context_noise) if is_context else float(dmd_timesteps[dmd_idx])
        is_last_dmd = dmd_idx >= len(dmd_timesteps) - 1

        def run(model, override=None):
            dit = model.component(expert_id)
            if is_context:
                # Clean-context pass: one forward at timestep=context_noise to write clean K/V into the
                # sliding-window cache (return value discarded). Faithful to _update_context_cache.
                if hasattr(dit, "call"):
                    self._eps(dit, block, st, start_frame, num_frames, float(self.context_noise), frame_seq_len)
                return StepResult(output={
                    "block": np.asarray(block, dtype="float32"),
                    "start": start_frame,
                    "num": num_frames,
                    "context": True
                })
            eps = self._eps(dit, block, st, start_frame, num_frames, timestep, frame_seq_len)
            if boundary_ts is not None and timestep >= boundary_ts:  # MoE high-noise expert (variants only)
                x0 = pred_noise_to_x_bound(eps, block, timestep, boundary_ts, sigmas, tsteps)
            else:
                x0 = pred_noise_to_pred_video(eps, block, timestep, sigmas, tsteps)
            if is_last_dmd:
                new_block = x0  # last DMD step: the clean x0 IS the block result
            else:
                next_t = float(dmd_timesteps[dmd_idx + 1])
                noise = rng.standard_normal(np.asarray(x0).shape)
                new_block = add_noise(x0, noise, next_t, sigmas, tsteps)  # re-noise to the next DMD timestep
            new_block = precision.cast(new_block)
            return StepResult(
                output={
                    "block": np.asarray(new_block, dtype="float32"),
                    "eps": np.asarray(eps, dtype="float32"),
                    "start": start_frame,
                    "num": num_frames,
                    "context": False
                })

        res = ResourceRequest(compute_seconds=self.cost.predict(int(np.prod(block.shape)), 1.0),
                              resident_bytes=int(x_full.nbytes),
                              peak_activation_bytes=int(block.nbytes))
        label = (f"matrixgame2.context.{b_idx}" if is_context else f"matrixgame2.denoise.{b_idx}.{dmd_idx}")
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.DIFFUSION_STEP,
                        shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                                 dims=tuple(block.shape),
                                                 dtype=precision.compute_dtype,
                                                 extra=(("cfg", type(self.cfg).__name__), ("dmd", True), ("phase",
                                                                                                          phase))),
                        resources=res,
                        payload={
                            "branch": "dmd",
                            "block": b_idx,
                            "dmd": dmd_idx,
                            "phase": phase
                        },
                        run=run,
                        label=label,
                        capturable=False)  # causal KV-cache mutation + host-RNG re-noise -> eager path

    # --------------------------------------------------------------------- #
    # advance                                                                #
    # --------------------------------------------------------------------- #
    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        # Monotonic progress counter: control flow runs off block_idx/dmd_idx/phase in scratch, but the
        # runtime's no-progress watchdog (engine ProgramRunner._progress) keys on st.step_idx. Bump it on
        # EVERY executed work unit (each DMD step AND each clean-context pass) so a multi-block causal
        # rollout is seen to advance.
        st.step_idx += 1
        out = result.output
        start, num = out["start"], out["num"]
        # Write the (denoised or unchanged) block back into the full latent.
        x_full = np.array(st.latents["video"])
        x_full[:, start:start + num] = out["block"]
        st.latents["video"] = x_full

        if out.get("context"):  # finished the clean-context pass -> advance to the next block
            if st.profile == ExecutionProfile.ROLLOUT:
                st.trajectory.append({"block": st.scratch["block_idx"], "latents": np.asarray(x_full).copy()})
            st.scratch["block_idx"] += 1
            st.scratch["dmd_idx"] = 0
            st.scratch["phase"] = "denoise"
            return st

        # Mid-block DMD step.
        dmd_idx = st.scratch["dmd_idx"]
        if dmd_idx >= len(st.scratch["dmd_timesteps"]) - 1:
            # Block fully DMD-denoised -> run the clean-context cache-refresh pass next.
            st.scratch["phase"] = "context"
        else:
            st.scratch["dmd_idx"] = dmd_idx + 1
        return st

    # --------------------------------------------------------------------- #
    # finalize                                                               #
    # --------------------------------------------------------------------- #
    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["video"]},
                          metrics={"denoise_blocks": float(len(st.scratch["blocks"]))},
                          behavior=st.trajectory or None)
