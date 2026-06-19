"""HYWorldDenoiseLoop — chunk-rollout flow-match denoise for the HY-WorldPlay world model.

The standard ``WanDenoiseLoop`` runs ONE flow-match sweep over the full latent with a scalar timestep.
HY-WorldPlay instead splits the latent into temporal CHUNKS (``chunk_latent_frames=16``) and rolls out
chunk by chunk (faithful to ``fastvideo/pipelines/stages/hyworld_denoising.py:HYWorldDenoisingStage``):

  * chunk 0 (always runs): a standard per-chunk flow-match sweep over the first 16 latent frames; the
    per-frame timestep is just the live ``t`` broadcast to all 16 frames.
  * chunk>0 (BRINGUP, see below): selects camera-aligned history "memory" frames via
    ``select_aligned_memory_frames`` (a 50k-point-sphere geometry op), prepends them as FROZEN context
    (timestep pinned at ``stabilization_level - 1 = 14``) with the live chunk's timestep ``t``, denoises
    only the current 16-frame slice, and writes that slice back.

The flow-match ODE math + ClassicCFG reuse v2's ``FLOW_MATCH_STEP`` kernel + ``ClassicCFG`` policy; the
chunk / context / dual-timestep orchestration is the new part. The DiT adapter (``HYWorldDiT``) marshals
the 65ch latent concat, the 3-stream conditioning list, the per-frame heterogeneous timestep, and the
camera/action tensors INTERNALLY, so this loop only calls the cosmos2/ToyDiT-compatible
``dit(latent, text_embed, sigma)`` (with the per-chunk camera context threaded via ``context=``).

BRINGUP (needs a request-API extension to be real):
  * The default preset (num_frames=125 -> ~32 latent frames, chunk 16 -> 2 chunks) will execute chunk>0.
    Camera-aligned memory retrieval, ProPE camera attention, and per-frame action conditioning all need
    ``viewmats``/``Ks``/``action`` carried from a pose string (``pose="w-31"`` -> ``pose_to_input`` +
    ``compute_latent_num``). The v2 request/program has no slot for the pose string yet, so absent a
    pose (the CPU toy path) this loop runs the degenerate per-chunk t2v path: it clamps ``chunk_num`` to
    the number of whole chunks but skips camera/memory selection (the adapter builds zero camera/action
    internally). On a GPU box with a pose-carrying request the chunk>0 branch (memory retrieval + frozen
    context) activates; that path is written-not-run here.
"""
from __future__ import annotations

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

# Real HY-WorldPlay VAE (AutoencoderKLHYWorld) compression: z=32, 4x temporal, 16x spatial.
HYWORLD_LATENT_CHANNELS = 32
HYWORLD_TEMPORAL_RATIO = 4
HYWORLD_SPATIAL_RATIO = 16
# Chunk rollout config (from HYWorldDenoisingStage + the preset defaults).
HYWORLD_CHUNK_LATENT_FRAMES = 16
HYWORLD_STABILIZATION_LEVEL = 15  # context (history) frames are pinned at stabilization_level - 1 = 14


class HYWorldDenoiseLoop:
    """Outer chunk loop x inner flow-match step loop. ``next``/``advance`` step the inner loop and roll
    the outer chunk index when a chunk's sweep finishes (all per-request state in ``LoopState``)."""

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 latent_channels: int = HYWORLD_LATENT_CHANNELS,
                 spatial_ratio: int = HYWORLD_SPATIAL_RATIO,
                 temporal_ratio: int = HYWORLD_TEMPORAL_RATIO,
                 chunk_latent_frames: int = HYWORLD_CHUNK_LATENT_FRAMES,
                 stabilization_level: int = HYWORLD_STABILIZATION_LEVEL):
        self.loop_id = loop_id
        self.cfg = cfg
        self.flow_shift = flow_shift
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio
        self.chunk_latent_frames = chunk_latent_frames
        self.stabilization_level = stabilization_level

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        # Preset overrides the schedule with explicit sigmas linspace(1.0, 0.0, 51)[:-1]; otherwise the
        # flow-shifted schedule. ``timesteps = sigma * 1000`` (num_train_timesteps=1000) per the convention.
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
                       timesteps=[float(s) * 1000.0 for s in sig])
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        # Chunk layout: the latent's temporal dim split into whole chunks of chunk_latent_frames. The toy
        # CPU latent is tiny (T may be < chunk size) -> clamp to a single chunk so the loop still exercises.
        n_frames = int(x.shape[1])
        chunk = max(1, min(self.chunk_latent_frames, n_frames))
        st.scratch["chunk_latent_frames"] = chunk
        st.scratch["chunk_num"] = max(1, n_frames // chunk)
        st.scratch["chunk_idx"] = 0
        # World-model conditioning (BRINGUP): the pose string -> viewmats/Ks/action expansion and the
        # camera-aligned memory retrieval need a request-API slot that v2 does not carry yet. Absent (CPU
        # toy / no pose) -> chunk>0 runs WITHOUT the frozen camera-aligned context (degenerate per-chunk
        # sweep). The adapter builds zero camera/action internally so the DiT call is unchanged.
        st.scratch["camera_ctx"] = ctx.slots.get("camera_ctx")  # {viewmats, Ks, action, points_local} | None
        # I2V (BRINGUP): SigLIP image embeds + the [cond_latent | mask] (33ch) latent; None for t2v -> the
        # adapter passes zero image embeds + a zero cond latent (the DiT detects all-zero image -> masks it).
        st.scratch["i2v_img_embeds"] = ctx.slots.get("i2v_img_embeds")
        st.scratch["i2v_cond"] = ctx.slots.get("i2v_cond")
        st.plugin_state["cfg"] = {}
        return st

    def next(self, st: LoopState):
        # Inner step index walks the sigma schedule; when it reaches the end, roll to the next chunk.
        i = st.step_idx
        n_steps = len(st.sigmas) - 1
        chunk_idx = st.scratch["chunk_idx"]
        if chunk_idx >= st.scratch["chunk_num"]:
            return Done()
        if i >= n_steps:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        expert_id = self.expert.expert_for(StepContext(i, st.timesteps[i], sigma_t))
        sctx = StepContext(step_idx=i, timestep=st.timesteps[i], sigma=sigma_t, active_expert_id=expert_id)
        cfg_state = st.plugin_state["cfg"]
        branches = self.cfg.branches_this_step(sctx, cfg_state)
        chunk = st.scratch["chunk_latent_frames"]
        start = chunk_idx * chunk
        end = start + chunk
        # The current chunk's latent slice (the only frames this chunk denoises + writes back).
        x_full = st.latents["video"]
        x_chunk = np.asarray(x_full[:, start:end], dtype="float32")
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision
        # Per-chunk camera/action context threaded to the adapter (None on the CPU/degenerate path). The
        # adapter is responsible for slicing it to [start:end] and prepending memory frames (BRINGUP).
        cam_ctx = st.scratch.get("camera_ctx")
        i2v_ctx, i2v_cond = st.scratch.get("i2v_img_embeds"), st.scratch.get("i2v_cond")
        # The conditioning the DiT call carries: a per-chunk dict (camera/action/i2v) the adapter unpacks.
        # Kept as a single ``context=`` arg so the loop's dit-call stays ToyDiT-compatible (cosmos2 rule).
        chunk_context = None
        if cam_ctx is not None or i2v_ctx is not None or i2v_cond is not None:
            chunk_context = {
                "camera": cam_ctx,
                "chunk": (start, end),
                "chunk_idx": chunk_idx,
                "i2v_img_embeds": i2v_ctx,
                "i2v_cond": i2v_cond,
                "stabilization_level": self.stabilization_level
            }

        def _velocity(model: object, x_: np.ndarray, sigma_t_: float, pe_: object, ne_: object,
                      scale_: float) -> np.ndarray:
            # Conditioned forward + ClassicCFG combine. The pos/neg branches differ only in the text embed
            # (the adapter swaps the matching encoder_attention_mask internally). The chunk's camera/action
            # conditioning rides in ``context``; None -> the DiT runs the plain (degenerate) chunk forward.
            dit = model.component(expert_id)  # type: ignore[attr-defined]
            preds = {b: dit(x_, pe_ if b == "cond" else ne_, sigma_t_, context=chunk_context) for b in branches}
            return precision.cast(cfg.combine(preds, scale_, sctx, cfg_state))

        def run(model, override=None):
            kernels = model.platform.kernels
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                velocity = _velocity(model, x_chunk, sigma_t, pe, ne, scale)
            # Flow-match Euler on the CURRENT-CHUNK slice only (HYWorldDenoisingStage steps the chunk).
            x_next = kernels.get(FLOW_MATCH_STEP)(precision.cast(x_chunk), velocity, sigma_t, sigma_next)
            return StepResult(
                output={
                    "noise_pred": np.asarray(velocity, dtype="float32"),
                    "latents": x_next.astype("float32"),
                    "chunk": (start, end)
                })

        cond_bytes = sum(int(np.asarray(e).nbytes) for e in (pe, ne) if e is not None)
        res = ResourceRequest(compute_seconds=self.cost.predict(int(np.prod(x_chunk.shape)), float(len(branches))),
                              resident_bytes=int(x_full.nbytes) + cond_bytes,
                              peak_activation_bytes=int(x_chunk.nbytes))
        return WorkPlan(
            loop_id=self.loop_id,
            instance_id=st.instance_id,
            kind=WorkUnitKind.DIFFUSION_STEP,
            shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                     dims=tuple(x_chunk.shape),
                                     dtype=precision.compute_dtype,
                                     extra=(("cfg", type(cfg).__name__), ("chunk", chunk_idx))),
            resources=res,
            payload={
                "branch": "combined",
                "step": i,
                "chunk": chunk_idx
            },
            run=run,
            label=f"hyworld.denoise.c{chunk_idx}.{i}",
            # Chunk rollout + camera-aligned memory retrieval + heterogeneous per-frame timestep -> eager.
            capturable=False)

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        # Write the denoised chunk slice back into the full latent (only this chunk's frames change).
        start, end = result.output["chunk"]
        x_full = np.asarray(st.latents["video"], dtype="float32")
        x_full[:, start:end] = np.asarray(result.output["latents"], dtype="float32")
        st.latents["video"] = x_full
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({
                "step": st.step_idx,
                "chunk": st.scratch["chunk_idx"],
                "sigma": st.sigmas[st.step_idx],
                "velocity": np.asarray(result.output["noise_pred"]).copy(),
                "latents": np.asarray(st.latents["video"]).copy()
            })
        st.step_idx += 1
        if st.step_idx >= len(st.sigmas) - 1:  # this chunk's flow-match sweep finished -> next chunk
            st.scratch["chunk_idx"] += 1
            st.step_idx = 0
            st.plugin_state["cfg"] = {}  # reset per-chunk CFG state
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["video"]},
                          metrics={"chunks": float(st.scratch["chunk_num"])},
                          behavior=st.trajectory or None)
