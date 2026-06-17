"""ChunkRolloutLoop — causal/AR video (design_v3 §4 causal, §5; design.md taxonomy).

Loop shape: outer loop over latent-frame chunks × inner denoise per chunk, carrying a per-chunk
KV/context via the slab-KV cache (sink/local window in inference; full history in self-forcing
training mode). Each completed chunk is streamable (``StepResult.emit``), and the rollout profile
captures per-chunk behavior for the self-forcing distillation method (design_v3 §10).

State is entirely in ``LoopState`` (the slab list + chunk position), so interleaving causal
rollouts of two sessions cannot smear context — and the slab-KV pool is namespaced per request.
"""
from __future__ import annotations

import numpy as np

from ..._enums import ExecutionProfile, WorkUnitKind
from ...cache.classes import Slab
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
from ...loop.sampler import flow_match_euler_step
from ...request.streams import StreamChunk
from ..backend import LATENT_CHANNELS


class ChunkRolloutLoop:
    def __init__(self, *, loop_id, num_chunks, chunk_size, steps_per_chunk, cfg, flow_shift,
                 precision, cost):
        self.loop_id = loop_id
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.steps_per_chunk = steps_per_chunk
        self.cfg = cfg
        self.flow_shift = flow_shift
        self.precision = precision
        self.cost = cost

    def _chunk_shape(self, req) -> tuple[int, int, int, int]:
        h = max(2, req.diffusion.height // 120)
        w = max(2, req.diffusion.width // 120)
        return (LATENT_CHANNELS, self.chunk_size, h, w)

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        sig = self.flow_shift.build_schedule(self.steps_per_chunk, req.diffusion.height, req.diffusion.width)
        st = LoopState(loop_id=self.loop_id, instance_id=model.card.model_id,
                       request_id=req.request_id, profile=ctx.profile, rng=rng, seed=seed,
                       sigmas=[float(s) for s in sig], timesteps=[float(s) * 1000.0 for s in sig])
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        # world-model continuation: seed the chunk context from prior chunks (an interactive session's
        # persistent world state; design_v3 §16). Default [] ⇒ a fresh rollout (unchanged one-shot path).
        prior = [np.asarray(c, dtype="float32") for c in (ctx.slots.get("world_context") or [])]
        st.scratch.update(chunk_idx=0, step_in_chunk=0, slabs=prior, chunks_out=[],
                          guidance_scale=float(req.diffusion.guidance_scale),
                          caches=getattr(model, "caches", None))
        st.latents["chunk"] = (rng.standard_normal(self._chunk_shape(req)) * float(sig[0])).astype("float32")
        st.plugin_state["cfg"] = {}
        return st

    def next(self, st: LoopState):
        if st.scratch["chunk_idx"] >= self.num_chunks:
            return Done()
        i = st.scratch["step_in_chunk"]
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        sctx = StepContext(step_idx=i, timestep=st.timesteps[i], sigma=sigma_t)
        cfg_state = st.plugin_state["cfg"]
        branches = self.cfg.branches_this_step(sctx, cfg_state)
        x = st.latents["chunk"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        context = np.mean(st.scratch["slabs"], axis=0) if st.scratch["slabs"] else None
        cfg, precision = self.cfg, self.precision

        def run(model, override=None):
            if override is not None and "noise_pred" in override:
                velocity = np.asarray(override["noise_pred"], dtype="float32")
            else:
                dit = model.component("transformer")
                preds = {b: dit(x, pe if b == "cond" else ne, sigma_t, context=context) for b in branches}
                velocity = cfg.combine(preds, scale, sctx, cfg_state)
            x_next = flow_match_euler_step(precision.cast(x), precision.cast(velocity), sigma_t, sigma_next)
            return StepResult(output={"noise_pred": np.asarray(velocity, dtype="float32"),
                                      "latents": x_next.astype("float32")})

        emits = []
        if i == self.steps_per_chunk - 1:                       # last step of this chunk → streamable
            emits.append(StreamChunk(stream_id=st.request_id, modality="video",
                                     seq=st.scratch["chunk_idx"], data=x, preview=False))
        res = ResourceRequest(
            compute_seconds=self.cost.predict(int(np.prod(x.shape)), float(len(branches))),
            resident_bytes=int(x.nbytes), peak_activation_bytes=int(x.nbytes * len(branches)))
        return WorkPlan(
            loop_id=self.loop_id, instance_id=st.instance_id, kind=WorkUnitKind.CHUNK_STEP,
            shape_sig=ShapeSignature(WorkUnitKind.CHUNK_STEP, dims=tuple(x.shape),
                                     extra=(("chunk", st.scratch["chunk_idx"]),)),
            resources=res, payload={"branch": "combined", "chunk": st.scratch["chunk_idx"], "step": i},
            run=run, label=f"causal.c{st.scratch['chunk_idx']}.s{i}", emits=emits)

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["chunk"] = result.output["latents"]
        st.scratch["step_in_chunk"] += 1
        st.step_idx += 1
        if st.scratch["step_in_chunk"] >= self.steps_per_chunk:
            chunk = np.asarray(st.latents["chunk"]).copy()
            st.scratch["slabs"].append(chunk)                   # context for the next chunk
            st.scratch["chunks_out"].append(chunk)
            caches = st.scratch.get("caches")
            if caches is not None and caches.has("slab_kv"):     # demonstrate the slab-KV class
                caches.pool("slab_kv").append(st.request_id, Slab(st.scratch["chunk_idx"], k=chunk, v=None))
            if st.profile == ExecutionProfile.ROLLOUT:
                st.trajectory.append({"chunk": st.scratch["chunk_idx"], "latents": chunk})
            st.scratch["chunk_idx"] += 1
            st.scratch["step_in_chunk"] = 0
            if st.scratch["chunk_idx"] < self.num_chunks:
                st.latents["chunk"] = (st.rng.standard_normal(chunk.shape) * float(st.sigmas[0])).astype("float32")
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        video = np.concatenate(st.scratch["chunks_out"], axis=1) if st.scratch["chunks_out"] else None
        return LoopResult(outputs={"latents": video, "chunks": list(st.scratch["chunks_out"])},
                          metrics={"chunks": float(st.scratch["chunk_idx"])},
                          behavior=st.trajectory or None)
