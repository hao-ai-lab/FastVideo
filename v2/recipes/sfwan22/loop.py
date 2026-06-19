"""SFWan22ChunkRolloutLoop — Self-Forcing Wan2.2-A14B causal DMD rollout.

Self-Forcing Wan2.2-A14B is the CAUSAL student of the Wan2.2 MoE: a few-step DMD denoiser run
block-by-block (causal chunks, carrying per-chunk KV context) over TWO transformer experts switched by
a boundary timestep. This loop combines:

  * the causal/AR chunk structure of ``v2.recipes.wan_causal.ChunkRolloutLoop`` (outer loop over
    latent-frame blocks × inner denoise per block, each completed block streamable, behavior captured
    under the ROLLOUT profile for the self-forcing distillation method); with
  * the Wan2.2 MoE boundary routing (two experts, ``boundary_ratio``) and the DMD few-step math of
    ``fastvideo/pipelines/stages/causal_denoising.py:CausalDMDDenosingStage`` — ported faithfully:
      - DMD timesteps = the (warped) ``dmd_denoising_steps`` schedule in [0,1000] timestep space;
      - per DMD step, route by ``t_cur >= boundary_timestep`` (``boundary_ratio·1000``): high-noise expert
        ``transformer`` above the boundary, low-noise ``transformer_2`` below it;
      - convert the predicted noise to a clean video: ``pred_noise_to_x_bound`` for high-noise steps
        (``x = x_in - (σ_t - σ_boundary)·ε``), else ``pred_noise_to_pred_video`` (``x = x_in - σ_t·ε``);
      - re-noise to the next DMD timestep with ``add_noise_high`` (within the high-noise band) or
        ``add_noise`` (interpolation ``(1-σ)·x0 + σ·ε``); the last step keeps the clean ``x0``;
      - the MoE first block is 1 latent frame (the ``block_sizes[0]=1`` Wan2.2-MoE hack), and the final
        ``num_frames_per_block-1`` trailing frames are trimmed.

The sigma schedule reproduces ``SelfForcingFlowMatchScheduler``: shifted linspace ``σ_max→σ_min`` over
``num_train_timesteps`` so the DMD timestep→sigma lookup matches the GPU scheduler's ``argmin`` mapping.

i2v (optional): the program writes ``i2v_cond`` (first-frame VAE latent) + ``i2v_img_embeds`` (CLIP);
the first latent block is then the conditioning frame (priming both experts' KV) and is held fixed —
absent for t2v, so the loop degrades to pure causal t2v exactly as the fastvideo stage does.

The dit-call (``dit(x, text_embed, sigma, context=..., cond=...)``) is the same toy-compatible signature
the Wan/cosmos loops use, so this runs on the CPU ``ToyDiT`` and swaps to the GPU CausalWan adapter with
no loop change. All per-request state lives in ``LoopState`` (interleave-safe).
"""
from __future__ import annotations

import numpy as np

from v2._enums import ExecutionProfile, WorkUnitKind
from v2.cache.classes import Slab
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
from v2.platform.backends.toy import LATENT_CHANNELS
from v2.request.streams import StreamChunk

# Real Wan2.2 VAE (AutoencoderKLWan) compression: z_dim=16, 4x temporal, 8x spatial — same as Wan2.1.
WAN_LATENT_CHANNELS = 16
WAN_TEMPORAL_RATIO = 4
WAN_SPATIAL_RATIO = 8
NUM_TRAIN_TIMESTEPS = 1000  # SelfForcingFlowMatchScheduler.num_train_timesteps


def build_self_forcing_sigmas(num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
                              shift: float = 12.0,
                              sigma_max: float = 1.0,
                              sigma_min: float = 0.003 / 1.002) -> np.ndarray:
    """The shifted σ schedule of ``SelfForcingFlowMatchScheduler`` (one σ per train timestep).

    ``set_timesteps(num_inference_steps=num_train_timesteps)`` builds a linspace ``σ_max→σ_min`` of that
    length, then applies flow-shift ``σ' = shift·σ / (1 + (shift-1)·σ)``. ``timesteps = σ·1000``. The DMD
    loop's ``warp_denoising_step`` and the noise/x0 conversions both index this table by an ``argmin``
    nearest-timestep lookup, so reproducing it here makes the CPU rollout faithful to the GPU stage.
    """
    base = np.linspace(sigma_max, sigma_min, num_train_timesteps, dtype=np.float64)
    return shift * base / (1.0 + (shift - 1.0) * base)


def _warp_dmd_timesteps(dmd_steps: list[int], sigmas: np.ndarray) -> np.ndarray:
    """``warp_denoising_step``: map the integer DMD steps onto the scheduler timestep grid.

    The fastvideo stage does ``scheduler_timesteps = cat(timesteps, [0]); timesteps[1000 - dmd_steps]``
    where ``scheduler.timesteps = sigmas·1000``. We reproduce that index gather (clamped) so the warped
    DMD timesteps land on the same shifted grid the σ lookups use.
    """
    sched_t = np.concatenate([sigmas * NUM_TRAIN_TIMESTEPS, np.array([0.0])])  # len = num_train+1
    idx = np.clip(NUM_TRAIN_TIMESTEPS - np.asarray(dmd_steps, dtype=np.int64), 0, len(sched_t) - 1)
    return sched_t[idx]


def _sigma_at(timestep: float, sigmas: np.ndarray) -> float:
    """Nearest-timestep → σ lookup (the scheduler's ``argmin |timesteps - t|``)."""
    grid: np.ndarray = sigmas * NUM_TRAIN_TIMESTEPS
    return float(sigmas[int(np.argmin(np.abs(grid - float(timestep))))])


def latent_shape(req,
                 model=None,
                 *,
                 channels=WAN_LATENT_CHANNELS,
                 spatial_ratio=WAN_SPATIAL_RATIO,
                 frames=1) -> tuple[int, int, int, int]:
    """Latent geometry for ONE causal block (``frames`` latent frames). Real Wan geometry on the cuda
    backend; a tiny deterministic stand-in on the CPU toy (the heavy DiT/VAE only exist on GPU)."""
    d = req.diffusion
    if model is not None and getattr(getattr(model, "platform", None), "device", "cpu") == "cuda":
        return (channels, max(1, frames), max(1, d.height // spatial_ratio), max(1, d.width // spatial_ratio))
    return (LATENT_CHANNELS, max(1, frames), max(2, d.height // 120), max(2, d.width // 120))


class SFWan22ChunkRolloutLoop:
    """Causal DMD rollout for Self-Forcing Wan2.2-A14B (MoE + boundary + optional i2v first frame)."""

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 dmd_denoising_steps: list[int],
                 boundary_ratio: float = 0.875,
                 num_frames_per_block: int = 7,
                 warp_denoising_step: bool = True,
                 latent_channels: int = WAN_LATENT_CHANNELS,
                 spatial_ratio: int = WAN_SPATIAL_RATIO,
                 temporal_ratio: int = WAN_TEMPORAL_RATIO):
        self.loop_id = loop_id
        self.cfg = cfg  # carried for the WorkPlan op-structure key (DMD is few-step here)
        self.flow_shift = flow_shift
        self.precision = precision
        self.expert = expert  # BoundaryTimestepRouting (kept for the contract/op-structure key)
        self.cost = cost
        self.dmd_denoising_steps = list(dmd_denoising_steps)
        self.boundary_ratio = boundary_ratio
        self.num_frames_per_block = int(num_frames_per_block)
        self.warp_denoising_step = bool(warp_denoising_step)
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio

    # --- block layout (uniform causal blocks for the train-forward adapter path) ------------------- #
    def _block_sizes(self, has_image: bool) -> list[int]:
        """latent frames per causal block.

        The fastvideo ``CausalDMDDenosingStage`` runs a 1-latent-frame MoE head block then full
        ``num_frame_per_block`` blocks, carrying cross-chunk causal context through the transformer's
        INTERNAL kv_cache (``_forward_inference``). The v2 GPU ``WanDiT`` adapter, however, drives the
        kv_cache-LESS ``_forward_train`` path (the loop math is numpy fp32; no kv_cache is threaded), so:
          * each DiT call is independent — there is no real cross-chunk KV carryover to preserve, so the
            1-frame MoE head block (which only mattered to PRIME that kv_cache) is unnecessary; and
          * ``_forward_train`` caches ``self.block_mask`` on the FIRST call keyed to that call's token
            count, so VARYING the per-block frame count (the old ``[1, nfpb, nfpb]``) makes the cached
            flex-attention mask mismatch later blocks. Keeping every block the SAME size keeps the cached
            mask valid for the whole rollout.
        We therefore emit UNIFORM ``nfpb``-frame blocks. The trailing ``nfpb-1`` frames are still trimmed
        at finalize (matching the stage's final crop). For i2v the conditioning frame primes context but
        is not a denoise block here (the adapter ignores it on the train path), so the layout is the same.
        """
        nfpb = self.num_frames_per_block
        return [nfpb, nfpb]  # two uniform causal chunks (one full chunk + headroom for the trailing trim)

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        # The full shifted σ table (one σ per train timestep) and the warped DMD timestep schedule.
        sched_sigmas = build_self_forcing_sigmas(
            shift=self.flow_shift.shift_for(req.diffusion.height, req.diffusion.width))
        dmd_t = (_warp_dmd_timesteps(self.dmd_denoising_steps, sched_sigmas)
                 if self.warp_denoising_step else np.asarray(self.dmd_denoising_steps, dtype=np.float64))
        i2v_cond = ctx.slots.get("i2v_cond")
        has_image = i2v_cond is not None
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       sigmas=[float(s) for s in sched_sigmas],
                       timesteps=[float(t) for t in dmd_t])
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        boundary_t = self.boundary_ratio * NUM_TRAIN_TIMESTEPS
        # world-model continuation: seed the chunk context from prior chunks (interactive sessions).
        prior = [np.asarray(c, dtype="float32") for c in (ctx.slots.get("world_context") or [])]
        st.scratch.update(sched_sigmas=sched_sigmas,
                          dmd_timesteps=dmd_t,
                          boundary_timestep=boundary_t,
                          block_sizes=self._block_sizes(has_image),
                          block_idx=0,
                          step_in_block=0,
                          slabs=prior,
                          chunks_out=[],
                          guidance_scale=float(req.diffusion.guidance_scale),
                          caches=getattr(model, "caches", None),
                          stream_video=bool(req.outputs.stream.get("video")),
                          i2v_cond=i2v_cond,
                          i2v_img_embeds=ctx.slots.get("i2v_img_embeds"),
                          has_image=has_image)
        # The first block's latent. For an i2v conditioning frame the program supplies the encoded
        # first-frame latent; otherwise initialize at randn·σ_max (the SelfForcing init scale).
        first_frames = st.scratch["block_sizes"][0] if st.scratch["block_sizes"] else self.num_frames_per_block
        st.latents["block"] = self._init_block_latent(req, model, rng, sched_sigmas, first_frames)
        st.plugin_state["cfg"] = {}
        # prime the causal KV context from the i2v conditioning frame (both experts) before any denoise —
        # the stage's "fill the low/high noise kv cache with first_frame_latent at timestep 0" priming.
        # It only seeds context (NOT chunks_out): on the GPU path the conditioning latent IS written as
        # output frame 0; the program/decoder reattaches it there. Keeping the output stream to the
        # denoised blocks keeps the toy latent channel count homogeneous (the structural CPU check).
        if has_image:
            st.scratch["i2v_first_frame"] = np.asarray(i2v_cond, dtype="float32")
            st.scratch["slabs"].append(np.asarray(i2v_cond, dtype="float32"))
        return st

    def _init_block_latent(self, req, model, rng, sched_sigmas, frames) -> np.ndarray:
        shape = latent_shape(req, model, channels=self.latent_channels, spatial_ratio=self.spatial_ratio, frames=frames)
        return (rng.standard_normal(shape) * float(sched_sigmas[0])).astype("float32")

    def next(self, st: LoopState):
        blocks = st.scratch["block_sizes"]
        if st.scratch["block_idx"] >= len(blocks):
            return Done()
        i = st.scratch["step_in_block"]
        dmd_t = st.scratch["dmd_timesteps"]
        sched_sigmas = st.scratch["sched_sigmas"]
        boundary_t = st.scratch["boundary_timestep"]
        t_cur = float(dmd_t[i])
        sigma_t = _sigma_at(t_cur, sched_sigmas)
        # Boundary routing (timestep space, faithful to the stage): t_cur >= boundary -> high-noise expert.
        high_noise = t_cur >= boundary_t
        expert_id = self.expert.high_noise if high_noise else self.expert.low_noise
        sctx = StepContext(step_idx=i, timestep=t_cur, sigma=sigma_t, active_expert_id=expert_id)
        cfg_state = st.plugin_state["cfg"]
        branches = self.cfg.branches_this_step(sctx, cfg_state)
        x = st.latents["block"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision
        # causal context (the KV the prior clean chunks contribute) + i2v CLIP embeds. The real CausalWan
        # adapter holds per-block KV slabs (variable frame counts); the toy DiT only reads ``mean(context)``,
        # so we summarize each prior slab by its mean (variable shapes -> a 1D vector) — interleave-safe.
        context = (np.array([float(np.mean(s))
                             for s in st.scratch["slabs"]], dtype="float32") if st.scratch["slabs"] else None)
        i2v_ctx = st.scratch.get("i2v_img_embeds")
        n_dmd = len(dmd_t)
        # σ at the boundary timestep + at the NEXT dmd step (for the re-noise back to that step).
        sigma_boundary = _sigma_at(boundary_t, sched_sigmas)
        t_next = float(dmd_t[i + 1]) if i + 1 < n_dmd else None
        sigma_next = _sigma_at(t_next, sched_sigmas) if t_next is not None else 0.0
        # how many of the DMD steps are in the high-noise band (to switch the re-noise rule, as in the stage)
        n_high = int(np.sum(dmd_t >= boundary_t))

        def _x0_from_noise(x_in: np.ndarray, velocity: np.ndarray) -> np.ndarray:
            # pred_noise -> clean video. High-noise band uses the x_bound conversion (subtract the
            # boundary-relative sigma), else the plain flow conversion (x0 = x - σ·ε).
            if high_noise:
                return x_in - (sigma_t - sigma_boundary) * velocity
            return x_in - sigma_t * velocity

        def run(model, override=None):
            dit = model.component(expert_id)
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                preds = {b: dit(x, pe if b == "cond" else ne, sigma_t, context=context, cond=i2v_ctx) for b in branches}
                velocity = precision.cast(cfg.combine(preds, scale, sctx, cfg_state))
            x0 = _x0_from_noise(np.asarray(x, dtype="float32"), np.asarray(velocity, dtype="float32"))
            if t_next is None:
                x_next = x0  # last DMD step: keep the clean x0
            elif high_noise and i < n_high - 1:  # re-noise within the high-noise band
                x_next = self._add_noise_high(x0, st.rng, sigma_next, sigma_boundary)
            elif high_noise and i == n_high - 1:  # last high-noise step: hand x0 to low-noise
                x_next = x0
            else:  # low-noise band: standard flow interpolation
                x_next = (1.0 - sigma_next) * x0 + sigma_next * st.rng.standard_normal(x0.shape).astype("float32")
            return StepResult(
                output={
                    "noise_pred": np.asarray(velocity, dtype="float32"),
                    "latents": np.asarray(x_next, dtype="float32"),
                    "x0": np.asarray(x0, dtype="float32")
                })

        emits = []
        if i == n_dmd - 1:  # block fully denoised -> streamable
            emits.append(
                StreamChunk(stream_id=st.request_id,
                            modality="video",
                            seq=st.scratch["block_idx"],
                            data=x,
                            preview=False))
        res = ResourceRequest(compute_seconds=self.cost.predict(int(np.prod(x.shape)), float(len(branches))),
                              resident_bytes=int(x.nbytes),
                              peak_activation_bytes=int(x.nbytes * len(branches)))
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.CHUNK_STEP,
                        shape_sig=ShapeSignature(WorkUnitKind.CHUNK_STEP,
                                                 dims=tuple(x.shape),
                                                 dtype=precision.compute_dtype,
                                                 extra=(("cfg", type(cfg).__name__), ("expert", expert_id),
                                                        ("block", st.scratch["block_idx"]))),
                        resources=res,
                        payload={
                            "branch": "combined",
                            "block": st.scratch["block_idx"],
                            "step": i
                        },
                        run=run,
                        label=f"sfwan22.b{st.scratch['block_idx']}.s{i}",
                        emits=emits,
                        capturable=False)  # DMD re-noise has host RNG -> eager path

    def _add_noise_high(self, x0: np.ndarray, rng, sigma: float, sigma_boundary: float) -> np.ndarray:
        """``add_noise_high``: x = α·x0 + β·ε with α=(1-σ)/(1-σ_b), β=sqrt(σ²-(α·σ_b)²) (SelfForcing)."""
        alpha = (1.0 - sigma) / max(1.0 - sigma_boundary, 1e-6)
        beta = float(np.sqrt(max(sigma**2 - (alpha * sigma_boundary)**2, 0.0)))
        noise = rng.standard_normal(x0.shape).astype("float32")
        return (alpha * x0 + beta * noise).astype("float32")

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["block"] = result.output["latents"]
        st.scratch["step_in_block"] += 1
        st.step_idx += 1
        if st.scratch["step_in_block"] >= len(st.scratch["dmd_timesteps"]):
            chunk = np.asarray(result.output["x0"], dtype="float32").copy()  # the clean block
            st.latents["block"] = chunk
            st.scratch["slabs"].append(chunk)  # clean context for the next causal block
            st.scratch["chunks_out"].append(chunk)
            caches = st.scratch.get("caches")
            if caches is not None and caches.has("slab_kv"):
                caches.pool("slab_kv").append(st.request_id, Slab(st.scratch["block_idx"], k=chunk, v=None))
            if st.profile == ExecutionProfile.ROLLOUT:
                st.trajectory.append({"block": st.scratch["block_idx"], "latents": chunk})
            st.scratch["block_idx"] += 1
            st.scratch["step_in_block"] = 0
            if st.scratch["block_idx"] < len(st.scratch["block_sizes"]):
                nf = st.scratch["block_sizes"][st.scratch["block_idx"]]
                shape = chunk.shape[:1] + (nf, ) + chunk.shape[2:]
                st.latents["block"] = (st.rng.standard_normal(shape) *
                                       float(st.scratch["sched_sigmas"][0])).astype("float32")
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        chunks = st.scratch["chunks_out"]
        video = np.concatenate(chunks, axis=1) if chunks else None
        # MoE: trim the trailing num_frames_per_block-1 frames (the stage's final crop) when present.
        if video is not None and self.expert.high_noise != self.expert.low_noise:
            trim = self.num_frames_per_block - 1
            if trim > 0 and video.shape[1] > trim:
                video = video[:, :-trim, :, :]
        return LoopResult(outputs={
            "latents": video,
            "chunks": list(chunks)
        },
                          metrics={"blocks": float(st.scratch["block_idx"])},
                          behavior=st.trajectory or None)
