"""LTX2DenoiseLoop — a custom-step driven loop for LTX-2's 2-stage pipeline.

LTX-2's guidance is braided (multi-pass: positive / negative / STG-perturbed, combined per the
LTX formula), so the step body is hand-written using samplers + the DiT as a library. It is
still a Loop (scheduled, admitted, observed, parity-gated). One ``LTX2DenoiseLoop`` is
instantiated per stage (base / refine) with that stage's distilled sigma schedule; both stages
bind the SAME ``transformer`` component (shared by reference — one instance, many loops).

Real LTX-2 distilled schedules (from the repo):
  base  : [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]   (8 steps)
  refine: [0.909375, 0.725, 0.421875, 0.0]                                          (3 steps)
The real DiT predicts ``denoised`` (x0); the CPU toy predicts velocity, integrated by the same
flow-match Euler step — the 2-stage structure, distilled schedules, and multi-pass guidance are
what this models faithfully.
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
    StepResult,
    WorkPlan,
)
from v2.platform import FLOW_MATCH_STEP
from v2.platform.backends.toy import LATENT_CHANNELS

BASE_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
REFINE_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]

# Real LTX-2 latent: 128 channels; CausalVideoAutoencoder ~32x spatial / 8x temporal. The base stage
# runs at HALF the full latent spatial res (stage-2 upsamples 2x), so divide by 64 (=32*2).
LTX2_LATENT_CHANNELS = 128
LTX2_SPATIAL_RATIO = 32
LTX2_TEMPORAL_RATIO = 8


def ltx_base_latent_shape(req, model=None, *, full_res: bool = False) -> tuple[int, int, int, int]:
    """Latent geometry for an LTX-2 denoise loop. The *distilled* base STAGE runs half-res (stage-2
    upsamples 2x) → spatial//64; the single-stage LTX-2 *base model* runs full-res → spatial//32
    (``full_res=True``). 128 channels, 8x temporal."""
    d = req.diffusion
    div = LTX2_SPATIAL_RATIO if full_res else (LTX2_SPATIAL_RATIO * 2)
    if model is not None and getattr(getattr(model, "platform", None), "device", "cpu") == "cuda":
        t = (max(1, d.num_frames) - 1) // LTX2_TEMPORAL_RATIO + 1
        return (LTX2_LATENT_CHANNELS, t, max(1, d.height // div), max(1, d.width // div))
    t = max(1, d.num_frames // 40)
    toy_div = 120 if full_res else 240
    return (LATENT_CHANNELS, t, max(2, d.height // toy_div), max(2, d.width // toy_div))


def base_flow_sigmas(num_steps: int, shift: float = 1.0) -> list[float]:
    """Many-step rectified-flow sigma schedule (1→0) for the non-distilled LTX-2 *base* model, with an
    optional resolution shift. (The distilled stages use the fixed few-step BASE_SIGMAS/REFINE_SIGMAS.)"""
    n = max(1, int(num_steps))
    s = np.linspace(1.0, 0.0, n + 1)
    if shift != 1.0:
        s = shift * s / (1.0 + (shift - 1.0) * s)
    return [float(x) for x in s]


class LTX2DenoiseLoop:

    def __init__(self,
                 *,
                 loop_id,
                 stage,
                 sigmas,
                 cfg_scale,
                 stg_scale,
                 cost,
                 input_slot=None,
                 seed_offset=0,
                 audio_input_slot=None,
                 full_res=False,
                 request_steps=False,
                 shift=1.0):
        self.loop_id = loop_id
        self.stage = stage  # "base" | "refine" | "single" (base model)
        self.sigmas = list(sigmas)
        self.cfg_scale = cfg_scale
        self.stg_scale = stg_scale
        self.cost = cost
        self.input_slot = input_slot  # None => fresh noise (base); slot name => read latents (refine)
        self.audio_input_slot = audio_input_slot  # joint A/V: where to read the threaded audio latent
        self.seed_offset = seed_offset
        # Single-stage LTX-2 base model: full-res latent + a request-driven many-step schedule (distilled
        # stages keep full_res=False + the fixed few-step self.sigmas).
        self.full_res = full_res
        self.request_steps = request_steps
        self.shift = shift

    def init(self, req, model, ctx) -> LoopState:
        seed = (req.diffusion.seed if req.diffusion.seed is not None else 0) + self.seed_offset
        rng = np.random.default_rng(seed)
        sig = base_flow_sigmas(req.diffusion.num_steps, self.shift) if self.request_steps else self.sigmas
        if self.input_slot is None:
            x = (rng.standard_normal(ltx_base_latent_shape(req, model, full_res=self.full_res)) *
                 float(sig[0])).astype("float32")
        else:
            x = np.asarray(ctx.slots[self.input_slot], dtype="float32")
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
        # joint A/V (T2VS): denoise an audio latent alongside the video, with per-modality guidance.
        # Gated on the request asking for audio ⇒ the T2V path below is byte-for-byte unchanged.
        want_audio = "audio" in getattr(req.outputs, "modalities", frozenset())
        gpm = req.diffusion.guidance_per_modality or {}
        st.scratch["want_audio"] = want_audio
        st.scratch["v_guidance"] = float(gpm.get("video", self.cfg_scale))  # == cfg_scale for T2V
        st.scratch["a_guidance"] = float(gpm.get("audio", self.cfg_scale))
        if want_audio:
            if self.audio_input_slot is None:
                a_len = max(2, req.diffusion.num_frames // 20)
                au = (rng.standard_normal((LATENT_CHANNELS, a_len, 1, 1)) * float(sig[0])).astype("float32")
            else:
                au = np.asarray(ctx.slots[self.audio_input_slot], dtype="float32")
            st.latents["audio"] = au
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        x = st.latents["video"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        v_guidance, stg_scale = st.scratch["v_guidance"], self.stg_scale  # per-modality video CFG
        want_audio, a_guidance = st.scratch.get("want_audio", False), st.scratch.get("a_guidance", 0.0)
        au = st.latents.get("audio")

        def run(model, override=None):
            fm = model.platform.kernels.get(FLOW_MATCH_STEP)  # solver dispatched per (device, arch)
            if override is not None and "noise_pred" in override:
                velocity = np.asarray(override["noise_pred"], dtype="float32")
            else:
                dit = model.component("transformer")
                pos = dit(x, pe, sigma_t)  # full conditioned pass
                neg = dit(x, ne, sigma_t)  # negative (CFG) pass
                # STG-perturbed pass (here: drop text conditioning) — the multi-pass braid
                ptb = dit(x, np.zeros_like(pe) if pe is not None else None, sigma_t)
                velocity = (pos + (v_guidance - 1.0) * (pos - neg) + stg_scale * (pos - ptb))
            x_next = fm(x, np.asarray(velocity, dtype="float32"), sigma_t, sigma_next)
            out = {"noise_pred": np.asarray(velocity, dtype="float32"), "latents": x_next.astype("float32")}
            if want_audio and au is not None:  # joint audio denoise, conditioned on video
                dit = model.component("transformer")
                a_pos = dit(au, pe, sigma_t, context=x)  # context=x ⇒ audio synced to the video
                a_neg = dit(au, ne, sigma_t, context=x)
                a_vel = a_pos + (a_guidance - 1.0) * (a_pos - a_neg)  # audio's OWN guidance scale
                out["audio_latents"] = fm(au, a_vel, sigma_t, sigma_next).astype("float32")
            return StepResult(output=out)

        res = ResourceRequest(
            compute_seconds=self.cost.predict(int(np.prod(x.shape)), 3.0),  # 3 passes
            resident_bytes=int(x.nbytes),
            peak_activation_bytes=int(x.nbytes * 3))
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.DIFFUSION_STEP,
                        shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                                 dims=tuple(x.shape),
                                                 extra=(("stage", self.stage), )),
                        resources=res,
                        payload={
                            "branch": "combined",
                            "step": i,
                            "stage": self.stage
                        },
                        run=run,
                        label=f"ltx2.{self.stage}.{i}")

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["video"] = result.output["latents"]
        if "audio_latents" in result.output:
            st.latents["audio"] = result.output["audio_latents"]
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({
                "step": st.step_idx,
                "stage": self.stage,
                "latents": np.asarray(st.latents["video"]).copy()
            })
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        outs = {"latents": st.latents["video"]}
        if st.scratch.get("want_audio") and "audio" in st.latents:
            outs["audio_latents"] = st.latents["audio"]  # threaded to the next stage / decoded
        return LoopResult(outputs=outs,
                          metrics={f"{self.stage}_steps": float(st.step_idx)},
                          behavior=st.trajectory or None)


# LTX-2 audio VAE latent geometry: 8 channels, 16 mel bins, ~25 latent frames/sec (16000/160/4).
LTX2_AUDIO_CHANNELS = 8
LTX2_AUDIO_MEL_BINS = 16


class LTX23DenoiseLoop:
    """LTX-2.3 single-stage JOINT audio+video denoise (distilled few-step). Unlike the 2.0 toy audio path
    (separate DiT calls), the real 2.3 cross-attends video<->audio in ONE DiT forward — which the v2 DiT
    adapter exposes as ``dit(video, video_text, sigma, audio_latent=, audio_text=) -> (v_vel, a_vel)`` —
    so both latents step together each step. Single-pass (distilled guidance≈1.0). Full-res video latent
    (//32) + a ``[8, T_aud, 16]`` audio latent. For T2V (no audio requested) it runs video-only."""

    def __init__(self,
                 *,
                 loop_id="ltx2_3",
                 sigmas=None,
                 cfg_scale=1.0,
                 stg_scale=0.0,
                 cost,
                 audio_channels=LTX2_AUDIO_CHANNELS,
                 audio_mel_bins=LTX2_AUDIO_MEL_BINS):
        self.loop_id = loop_id
        self.stage = "single"
        self.sigmas = list(sigmas) if sigmas else list(BASE_SIGMAS)  # tuned distilled few-step schedule
        self.cfg_scale = cfg_scale
        self.stg_scale = stg_scale
        self.cost = cost
        self.audio_channels = audio_channels
        self.audio_mel_bins = audio_mel_bins

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        sig = self.sigmas
        x = (rng.standard_normal(ltx_base_latent_shape(req, model, full_res=True)) * float(sig[0])).astype("float32")
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"video": x},
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) for s in sig])
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["audio_prompt_embeds"] = ctx.slots.get("audio_text_embeds")
        want_audio = "audio" in getattr(req.outputs, "modalities", frozenset())
        st.scratch["want_audio"] = want_audio
        if want_audio:
            t_aud = max(2, int(req.diffusion.num_frames))  # ~one audio latent frame per video frame
            au = (rng.standard_normal(
                (self.audio_channels, t_aud, self.audio_mel_bins)) * float(sig[0])).astype("float32")
            st.latents["audio"] = au
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        x = st.latents["video"]
        pe = st.cond["prompt_embeds"]
        apa = st.cond.get("audio_prompt_embeds")
        au = st.latents.get("audio")

        def run(model, override=None):
            fm = model.platform.kernels.get(FLOW_MATCH_STEP)
            dit = model.component("transformer")
            if au is not None:  # joint A/V: one forward -> both velocities
                vvel, avel = dit(x, pe, sigma_t, audio_latent=au, audio_text=apa)
                out = {
                    "latents": fm(x, np.asarray(vvel, "float32"), sigma_t, sigma_next).astype("float32"),
                    "audio_latents": fm(au, np.asarray(avel, "float32"), sigma_t, sigma_next).astype("float32"),
                    "noise_pred": np.asarray(vvel, dtype="float32")
                }
            else:  # video-only (T2V)
                vvel = dit(x, pe, sigma_t)
                out = {
                    "latents": fm(x, np.asarray(vvel, "float32"), sigma_t, sigma_next).astype("float32"),
                    "noise_pred": np.asarray(vvel, dtype="float32")
                }
            return StepResult(output=out)

        res = ResourceRequest(compute_seconds=self.cost.predict(int(np.prod(x.shape)), 1.0),
                              resident_bytes=int(x.nbytes),
                              peak_activation_bytes=int(x.nbytes * 2))
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.DIFFUSION_STEP,
                        shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                                 dims=tuple(x.shape),
                                                 extra=(("stage", self.stage), )),
                        resources=res,
                        payload={
                            "step": i,
                            "stage": self.stage
                        },
                        run=run,
                        label=f"ltx2_3.{i}")

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["video"] = result.output["latents"]
        if "audio_latents" in result.output:
            st.latents["audio"] = result.output["audio_latents"]
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({
                "step": st.step_idx,
                "stage": self.stage,
                "latents": np.asarray(st.latents["video"]).copy()
            })
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        outs = {"latents": st.latents["video"]}
        if st.scratch.get("want_audio") and "audio" in st.latents:
            outs["audio_latents"] = st.latents["audio"]
        return LoopResult(outputs=outs, metrics={"single_steps": float(st.step_idx)}, behavior=st.trajectory or None)
