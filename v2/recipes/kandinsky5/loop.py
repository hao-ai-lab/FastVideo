"""Kandinsky5DenoiseLoop — flow-match Euler over a CHANNELS-LAST video latent with a dual text stream.

Faithful port of the Kandinsky-5.0 T2V denoise loop (diffusers ``KandinskyV5...Pipeline.__call__``):
N flow-match Euler steps over the full latent, velocity prediction, ``ClassicCFG`` combine
(``uncond + s·(cond − uncond)``), σ·1000 timestep convention. It mirrors ``WanDenoiseLoop`` but
diverges on the three things Kandinsky needs (BRINGUP risks A/B/D from the port spec):

  1. **Channels-LAST DiT geometry, but a channels-FIRST loop latent.** The DiT forward consumes
     ``hidden_states[B, T, H, W, C]`` (C=4) and returns a channels-last velocity. The loop, however,
     keeps the latent channels-FIRST ``(c, t, h, w)`` — the same convention as Wan and the CPU
     ``ToyDiT`` (which channel-mixes over axis 0). The ``Kandinsky5DiT`` adapter does the
     channels-first → channels-last permute (and permutes the velocity back) INTERNALLY, so the loop
     and toy stay backend-agnostic and the GPU forward still sees the geometry it requires.
  2. **A SECOND conditioning slot.** Qwen2.5-VL token embeds go to one slot (the cross-attended text);
     the CLIP **pooled** vector goes to a second slot and is threaded to the dit-call as ``context`` (so
     the dit-call signature stays ``dit(latent, text_embed, sigma, context=pooled)`` — compatible with the
     CPU ``ToyDiT``). The ``Kandinsky5DiT`` adapter feeds ``context`` to ``pooled_projections``, which is
     MANDATORY (the forward raises if it is None).
  3. **Per-request RoPE positions + scale_factor are built INSIDE the adapter** from the latent geometry
     and resolution (``visual_rope_pos`` over the patched grid, ``text_rope_pos`` over the Qwen seq len,
     ``scale_factor`` = (1,2,2) for 480–854 else (1,3.16,3.16)). Keeping that in the adapter rather than
     threading extra kwargs is what lets the loop call the toy-compatible 3-arg dit signature.

The flow-match math (``FlowShiftPolicy.build_schedule`` → ``FLOW_MATCH_STEP``) and the CFG combine are
reused verbatim from the Wan recipe — they are identical to Kandinsky's. Streaming preview + the
ROLLOUT trajectory capture mirror ``WanDenoiseLoop``.
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
from v2.request.streams import StreamChunk

# AutoencoderKLHunyuanVideo compression for Kandinsky-5: z=16 latent channels, 8× spatial, 4× temporal.
# BRINGUP: the real Lite-5s checkpoint config is in_visual_dim/out_visual_dim/latent_channels = 16 (NOT 4 —
# 4 was a stale guess). The DiT's visual_cond=True path concats an extra cond+mask to 33 last-dim INSIDE the
# adapter; the loop/VAE latent stays the 16-channel diffusion latent.
KANDINSKY5_LATENT_CHANNELS = 16
KANDINSKY5_TEMPORAL_RATIO = 4
KANDINSKY5_SPATIAL_RATIO = 8


def latent_shape(req,
                 model=None,
                 *,
                 channels=KANDINSKY5_LATENT_CHANNELS,
                 spatial_ratio=KANDINSKY5_SPATIAL_RATIO,
                 temporal_ratio=KANDINSKY5_TEMPORAL_RATIO) -> tuple[int, int, int, int]:
    """CHANNELS-FIRST latent geometry ``(c, t, h, w)`` for the denoise loop (Wan/ToyDiT convention — the
    toy channel-mixes over axis 0). On the GPU backend the true geometry is ``c = 4``,
    ``t = (num_frames-1)//4 + 1``, ``h = H//8``, ``w = W//8``; the ``Kandinsky5DiT`` adapter permutes this
    to the channels-LAST ``[B, T, H, W, C]`` the forward requires. The CPU toy uses a tiny stand-in.
    ``model is None`` (cost estimation in tests) keeps the toy shape."""
    d = req.diffusion
    if model is not None and getattr(getattr(model, "platform", None), "device", "cpu") == "cuda":
        t = (max(1, d.num_frames) - 1) // temporal_ratio + 1
        return (channels, max(1, t), max(1, d.height // spatial_ratio), max(1, d.width // spatial_ratio))
    # toy stand-in: small, channels-first, with even h,w so the patch_size=(1,2,2) grid stays valid.
    t = max(1, d.num_frames // 40)
    h = max(2, d.height // 120)
    w = max(2, d.width // 120)
    return (LATENT_CHANNELS, t, h - (h % 2), w - (w % 2))


class Kandinsky5DenoiseLoop:

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 latent_channels=KANDINSKY5_LATENT_CHANNELS,
                 spatial_ratio=KANDINSKY5_SPATIAL_RATIO,
                 temporal_ratio=KANDINSKY5_TEMPORAL_RATIO):
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
        # Qwen token embeds (the cross-attended text stream) + the matching negative.
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        # CLIP pooled vectors (the MANDATORY pooled_projections, threaded to the dit-call as `context`).
        st.cond["pooled"] = ctx.slots.get("clip_pooled")
        st.cond["negative_pooled"] = ctx.slots.get("neg_clip_pooled")
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
        branches = self.cfg.branches_this_step(sctx, cfg_state)
        x = st.latents["video"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        pp, npp = st.cond["pooled"], st.cond["negative_pooled"]
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision

        def _velocity(model: Any, x_: Any, sigma_t_: float, pe_: Any, ne_: Any, pp_: Any, npp_: Any,
                      scale_: float) -> np.ndarray:
            # The conditioned forward + CFG combine. The CLIP pooled vector rides in ``context`` (the
            # toy ignores it harmlessly; the Kandinsky5DiT adapter routes it to pooled_projections and
            # builds visual/text RoPE + scale_factor internally from the latent geometry + resolution).
            dit = model.component(expert_id)
            preds = {
                b: dit(x_, pe_ if b == "cond" else ne_, sigma_t_, context=(pp_ if b == "cond" else npp_))
                for b in branches
            }
            return precision.cast(cfg.combine(preds, scale_, sctx, cfg_state))

        def run(model, override=None):
            kernels = model.platform.kernels
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                velocity = _velocity(model, x, sigma_t, pe, ne, pp, npp, scale)
            x_next = kernels.get(FLOW_MATCH_STEP)(precision.cast(x), velocity, sigma_t, sigma_next)
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": x_next.astype("float32")
            })

        def graph_fn(model, ws):
            # The CAPTURABLE op-structure: reads every per-step input from the static workspace.
            st_t, st_n = float(ws["sigma_t"]), float(ws["sigma_next"])
            velocity = _velocity(model, ws["x"], st_t, ws["pe"], ws["ne"], ws["pp"], ws["npp"], float(ws["scale"]))
            x_next = model.platform.kernels.get(FLOW_MATCH_STEP)(precision.cast(ws["x"]), velocity, st_t, st_n)
            np.copyto(ws["out"], x_next.astype("float32"))
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": np.array(ws["out"], copy=True)
            })

        cond_bytes = sum(int(np.asarray(e).nbytes) for e in (pe, ne, pp, npp) if e is not None)
        res = ResourceRequest(compute_seconds=self.cost.predict(int(np.prod(x.shape)), float(len(branches))),
                              resident_bytes=int(x.nbytes) + cond_bytes,
                              peak_activation_bytes=int(x.nbytes))
        emits = []
        if st.scratch.get("stream_video"):
            emits.append(StreamChunk(stream_id=st.request_id, modality="video", seq=i, data=x, preview=True))
        return WorkPlan(loop_id=self.loop_id,
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
                        label=f"kandinsky5.denoise.{i}",
                        emits=emits,
                        capturable=True,
                        graph_key=(tuple(sorted(branches)), expert_id, precision.scheduler_step_in_fp32),
                        graph_fn=graph_fn,
                        graph_inputs={
                            "x": x,
                            "sigma_t": sigma_t,
                            "sigma_next": sigma_next,
                            "pe": pe,
                            "ne": ne,
                            "pp": pp,
                            "npp": npp,
                            "scale": scale
                        })

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
