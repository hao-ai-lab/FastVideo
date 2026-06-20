"""ChunkRolloutLoop — causal/AR video.

Loop shape: outer loop over latent-frame chunks × inner denoise per chunk, carrying a per-chunk
KV/context via the slab-KV cache (sink/local window in inference; full history in self-forcing
training mode). Each completed chunk is streamable (``StepResult.emit``), and the rollout profile
captures per-chunk behavior for the self-forcing distillation method.

State is entirely in ``LoopState`` (the slab list + chunk position), so interleaving causal
rollouts of two sessions cannot smear context — and the slab-KV pool is namespaced per request.
"""
from __future__ import annotations

import numpy as np

from v2.core.enums import ExecutionProfile, WorkUnitKind
from v2.runtime.cache.classes import Slab
from v2.core.loop.contracts import (
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
from v2.core.request.streams import StreamChunk
from v2.platform.backends.toy import LATENT_CHANNELS


class ChunkRolloutLoop:

    def __init__(self, *, loop_id, num_chunks, chunk_size, steps_per_chunk, cfg, flow_shift, precision):
        self.loop_id = loop_id
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.steps_per_chunk = steps_per_chunk
        self.cfg = cfg
        self.flow_shift = flow_shift
        self.precision = precision

    def _chunk_shape(self, req, model=None) -> tuple[int, int, int, int]:
        # Real Wan geometry on the cuda backend (16 latent channels; chunk_size latent frames; 8x
        # spatial VAE compression); the CPU toy keeps a tiny stand-in.
        if model is not None and getattr(getattr(model, "platform", None), "device", "cpu") == "cuda":
            return (16, self.chunk_size, max(1, req.diffusion.height // 8), max(1, req.diffusion.width // 8))
        h = max(2, req.diffusion.height // 120)
        w = max(2, req.diffusion.width // 120)
        return (LATENT_CHANNELS, self.chunk_size, h, w)

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        sig = self.flow_shift.build_schedule(self.steps_per_chunk, req.diffusion.height, req.diffusion.width)
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) * 1000.0 for s in sig])
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        # world-model continuation: seed the chunk context from prior chunks (an interactive session's
        # persistent world state). Default [] ⇒ a fresh rollout (unchanged one-shot path).
        prior = [np.asarray(c, dtype="float32") for c in (ctx.slots.get("world_context") or [])]
        st.scratch.update(chunk_idx=0,
                          step_in_chunk=0,
                          slabs=prior,
                          chunks_out=[],
                          guidance_scale=float(req.diffusion.guidance_scale),
                          caches=getattr(model, "caches", None))
        st.latents["chunk"] = (rng.standard_normal(self._chunk_shape(req, model)) * float(sig[0])).astype("float32")
        st.plugin_state["cfg"] = {}
        # Real causal cross-chunk conditioning (cuda backend only): allocate the model's persistent
        # KV + cross-attn caches and own them HERE in LoopState (per request -> interleave-safe), then
        # thread them through every forward so the self-forcing student conditions each chunk on prior
        # clean chunks. Toy/CPU keeps the slab mean-context path (``context`` above). See loop.next.
        st.scratch["real_causal"] = False
        dit = model.component("transformer")
        if getattr(getattr(model, "platform", None), "device", "cpu") == "cuda" and hasattr(dit, "alloc_causal_caches"):
            _, _, lh, lw = self._chunk_shape(req, model)
            ps = getattr(getattr(dit, "module", None), "patch_size", (1, 2, 2))
            frame_seqlen = (lh // int(ps[-2])) * (lw // int(ps[-1]))
            caches = dit.alloc_causal_caches(frame_seqlen)
            st.scratch.update(causal_kv=caches["kv"],
                              causal_ca=caches["crossattn"],
                              frame_seqlen=frame_seqlen,
                              real_causal=True)
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
        # real-causal KV-cache threading (cuda backend); else the toy ``context`` path
        real_causal = st.scratch.get("real_causal", False)
        kv, ca = st.scratch.get("causal_kv"), st.scratch.get("causal_ca")
        fsl = st.scratch.get("frame_seqlen", 1560)
        ci = st.scratch["chunk_idx"]
        cur_start, start_frame = ci * self.chunk_size * fsl, ci * self.chunk_size
        is_last = (i == self.steps_per_chunk - 1)

        def run(model, override=None):
            fm = model.platform.kernels.get(FLOW_MATCH_STEP)  # solver dispatched per (device, arch)
            if override is not None and "noise_pred" in override:
                velocity = np.asarray(override["noise_pred"], dtype="float32")
            else:
                dit = model.component("transformer")
                if real_causal:  # condition on prior clean chunks via the model's kv_cache (CausVid Alg 2)
                    kw = dict(kv_cache=kv,
                              crossattn_cache=ca,
                              current_start=cur_start,
                              start_frame=start_frame,
                              frame_seqlen=fsl)
                    preds = {b: dit(x, pe if b == "cond" else ne, sigma_t, **kw) for b in branches}
                else:
                    preds = {b: dit(x, pe if b == "cond" else ne, sigma_t, context=context) for b in branches}
                velocity = cfg.combine(preds, scale, sctx, cfg_state)
            x_next = fm(precision.cast(x), precision.cast(velocity), sigma_t, sigma_next)
            if real_causal and is_last:  # write the clean chunk's K/V (timestep ~0) so the next chunk attends to it
                model.component("transformer")(x_next,
                                               pe,
                                               0.0,
                                               kv_cache=kv,
                                               crossattn_cache=ca,
                                               current_start=cur_start,
                                               start_frame=start_frame,
                                               frame_seqlen=fsl)
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": x_next.astype("float32")
            })

        emits = []
        if i == self.steps_per_chunk - 1:  # last step of this chunk → streamable
            emits.append(
                StreamChunk(stream_id=st.request_id,
                            modality="video",
                            seq=st.scratch["chunk_idx"],
                            data=x,
                            preview=False))
        res = ResourceRequest(
                              resident_bytes=int(x.nbytes),
                              peak_activation_bytes=int(x.nbytes * len(branches)))
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.CHUNK_STEP,
                        shape_sig=ShapeSignature(WorkUnitKind.CHUNK_STEP,
                                                 dims=tuple(x.shape),
                                                 extra=(("chunk", st.scratch["chunk_idx"]), )),
                        resources=res,
                        payload={
                            "branch": "combined",
                            "chunk": st.scratch["chunk_idx"],
                            "step": i
                        },
                        run=run,
                        label=f"causal.c{st.scratch['chunk_idx']}.s{i}",
                        emits=emits)

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["chunk"] = result.output["latents"]
        st.scratch["step_in_chunk"] += 1
        st.step_idx += 1
        if st.scratch["step_in_chunk"] >= self.steps_per_chunk:
            chunk = np.asarray(st.latents["chunk"]).copy()
            st.scratch["slabs"].append(chunk)  # context for the next chunk
            st.scratch["chunks_out"].append(chunk)
            caches = st.scratch.get("caches")
            if caches is not None and caches.has("slab_kv"):  # demonstrate the slab-KV class
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
        return LoopResult(outputs={
            "latents": video,
            "chunks": list(st.scratch["chunks_out"])
        },
                          metrics={"chunks": float(st.scratch["chunk_idx"])},
                          behavior=st.trajectory or None)
