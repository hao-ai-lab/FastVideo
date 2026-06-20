"""WanDenoiseLoop — the canonical driven denoise loop.

Bidirectional video diffusion: N flow-match steps over the full latent. Composed from policies
(CFG / flow-shift / precision / expert routing). ``next`` is kernel-free (it builds the
forward+combine+solver thunk); ``advance`` folds the result and, under the ROLLOUT profile,
captures a behavior slice — so the *same* loop serves and rolls out for RL. All per-request
state lives in ``LoopState`` (interleave-safe).
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
from v2.platform import FLOW_MATCH_STEP, FLOW_SDE_STEP
from v2.request.streams import StreamChunk
from v2.platform.backends.toy import LATENT_CHANNELS

# Real Wan2.1 VAE (AutoencoderKLWan) compression: z_dim=16, 4x temporal, 8x spatial.
WAN_LATENT_CHANNELS = 16
WAN_TEMPORAL_RATIO = 4
WAN_SPATIAL_RATIO = 8


def latent_shape(req,
                 model=None,
                 *,
                 channels=WAN_LATENT_CHANNELS,
                 spatial_ratio=WAN_SPATIAL_RATIO,
                 temporal_ratio=WAN_TEMPORAL_RATIO) -> tuple[int, int, int, int]:
    """Latent geometry for the denoise loop. On the real (GPU) backend the DiT/VAE require the true Wan
    geometry; the CPU toy uses a tiny deterministic stand-in. ``model is None`` (e.g. cost-estimation in
    tests) keeps the toy shape. Defaults are Wan2.1 (16 channels, 4x temporal, 8x spatial); Wan2.2-TI2V
    passes channels=48, spatial_ratio=16 (its higher-compression z_dim=48 VAE)."""
    d = req.diffusion
    if model is not None and getattr(getattr(model, "platform", None), "device", "cpu") == "cuda":
        t = (max(1, d.num_frames) - 1) // temporal_ratio + 1
        return (channels, max(1, t), max(1, d.height // spatial_ratio), max(1, d.width // spatial_ratio))
    t = max(1, d.num_frames // 40)
    h = max(2, d.height // 120)
    w = max(2, d.width // 120)
    return (LATENT_CHANNELS, t, h, w)


class WanDenoiseLoop:

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 latent_channels=WAN_LATENT_CHANNELS,
                 spatial_ratio=WAN_SPATIAL_RATIO,
                 temporal_ratio=WAN_TEMPORAL_RATIO):
        self.loop_id = loop_id
        self.cfg = cfg
        self.flow_shift = flow_shift
        self.precision = precision
        self.expert = expert
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
        # Seed the latent with numpy (same values on every device) then hand it to the platform's array
        # namespace: numpy on CPU (identity), a single upload to a resident device tensor on a GPU box.
        xp = model.platform.xp
        x = xp.from_host((rng.standard_normal(shape) * float(sig[0])).astype("float32"))
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
        st.scratch["xp"] = xp  # array namespace (numpy on CPU, torch-on-device on cuda)
        st.scratch["on_device"] = (getattr(model.platform, "device", "cpu") == "cuda")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        # I2V (optional): the program writes the CLIP image embeds + the [mask|cond] latent into slots;
        # the WanDiT adapter concats the cond (16->36ch) and passes the embeds as encoder_hidden_states_image.
        # Absent for T2V -> None -> the dit call + capture are unchanged.
        st.scratch["i2v_cond"] = ctx.slots.get("i2v_cond")
        st.scratch["i2v_img_embeds"] = ctx.slots.get("i2v_img_embeds")
        # FlowGRPO RL rollout: switch the sampler to SDE-with-logprob (else deterministic ODE serve).
        st.scratch["sde"] = bool(getattr(req.diffusion, "sde_rollout", False))
        st.scratch["sde_noise_scale"] = float(getattr(req.diffusion, "sde_noise_scale", 0.7))
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
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision
        xp, on_device = st.scratch["xp"], st.scratch.get("on_device", False)
        sde, noise_scale, rng = st.scratch.get("sde", False), st.scratch.get("sde_noise_scale", 0.7), st.rng

        i2v_ctx, i2v_cond = st.scratch.get("i2v_img_embeds"), st.scratch.get("i2v_cond")

        def _velocity(model: Any, x_: Any, sigma_t_: float, pe_: Any, ne_: Any, scale_: float) -> Any:
            # The conditioned forward + CFG combine. Solver/forward dispatch through the platform's
            # kernel table (numpy on CPU, the device kernel on a GPU/accel backend).
            # i2v_ctx/i2v_cond are None for T2V (the plain forward); for i2v the adapter concats the
            # conditioning latent and passes the CLIP embeds as encoder_hidden_states_image.
            dit = model.component(expert_id)
            preds = {
                b: dit(x_, pe_ if b == "cond" else ne_, sigma_t_, context=i2v_ctx, cond=i2v_cond)
                for b in branches
            }
            return precision.cast(cfg.combine(preds, scale_, sctx, cfg_state))

        def run(model, override=None):
            # The EAGER thunk: handles the override (cached-prediction) path and the stochastic SDE
            # rollout path — both eager-break under capture. Allocates freely.
            kernels = model.platform.kernels
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                velocity = _velocity(model, x, sigma_t, pe, ne, scale)
            if sde:  # FlowGRPO rollout: stochastic + log-prob
                noise = rng.standard_normal(x.shape)
                x_next, logp, _m, _s = kernels.get(FLOW_SDE_STEP)(precision.cast(x),
                                                                  velocity,
                                                                  sigma_t,
                                                                  sigma_next,
                                                                  noise=noise,
                                                                  noise_scale=noise_scale)
                return StepResult(
                    output={
                        "noise_pred": xp.to_f32(velocity),
                        "latents": xp.to_f32(x_next),
                        "sde_logprob": logp,
                        "prev": xp.to_f32(x)
                    })
            x_next = kernels.get(FLOW_MATCH_STEP)(precision.cast(x), velocity, sigma_t, sigma_next)
            return StepResult(output={"noise_pred": xp.to_f32(velocity), "latents": xp.to_f32(x_next)})

        def graph_fn(model, ws):
            # The CAPTURABLE op-structure: reads EVERY per-step input from the static workspace (never
            # from closure over per-step data) and writes into the static output buffer — so the same
            # captured graph replays correctly with rebound buffer contents.
            st_t, st_n = float(ws["sigma_t"]), float(ws["sigma_next"])
            velocity = _velocity(model, ws["x"], st_t, ws["pe"], ws["ne"], float(ws["scale"]))
            x_next = model.platform.kernels.get(FLOW_MATCH_STEP)(precision.cast(ws["x"]), velocity, st_t, st_n)
            np.copyto(ws["out"], x_next.astype("float32"))  # write the static output buffer
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": np.array(ws["out"], copy=True)
            })

        cond_bytes = sum(int(e.nbytes) for e in (pe, ne) if e is not None)  # .nbytes works on numpy + torch
        res = ResourceRequest(
            resident_bytes=int(x.nbytes) + cond_bytes,  # latents + conditioning held for the loop
            peak_activation_bytes=int(x.nbytes))  # one step's transient working buffer
        emits = []
        if st.scratch.get("stream_video"):
            emits.append(StreamChunk(stream_id=st.request_id, modality="video", seq=i, data=xp.to_host(x),
                                     preview=True))  # carry the latent as a (host) preview payload
        return WorkPlan(
            loop_id=self.loop_id,
            instance_id=st.instance_id,
            kind=WorkUnitKind.DIFFUSION_STEP,
            shape_sig=ShapeSignature(
                WorkUnitKind.DIFFUSION_STEP,
                dims=tuple(x.shape),
                dtype=precision.compute_dtype,  # compute dtype is part of the key
                extra=(("cfg", type(cfg).__name__), )),
            resources=res,
            payload={
                "branch": "combined",
                "step": i
            },
            run=run,
            label=f"wan.denoise.{i}",
            emits=emits,
            # CUDA-graph capture (Path A): the deterministic ODE step is capturable; the stochastic
            # SDE rollout step has host RNG / a data-dependent branch, so it must eager-break. The
            # op-structure key carries the CFG branch set, active expert, and the scheduler-precision
            # flag so a step with a different branch set / expert / solver precision never replays an
            # incompatible captured graph. (Compute dtype rides in shape_sig.dtype above.) NOTE: this
            # is sound for branch-set/expert-determined policies (ClassicCFG); a policy whose op
            # structure forks on other state (PerModalityCFG modality) would need a graph_key hook.
            # i2v threads non-workspace conditioning -> eager; on-device uses the eager resident path
            # (the numpy static-workspace capture form below is the CPU/accel path)
            capturable=not sde and i2v_cond is None and not on_device,
            graph_key=(tuple(sorted(branches)), expert_id, precision.scheduler_step_in_fp32),
            # the static-buffer capture form: graph_fn reads all per-step inputs from the workspace
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
            to_host = st.scratch["xp"].to_host  # device tensor -> host numpy for the captured trajectory
            i = st.step_idx
            rec = {
                "step": i,
                "sigma": st.sigmas[i],
                "velocity": to_host(result.output["noise_pred"]).copy(),
                "latents": to_host(st.latents["video"]).copy()
            }
            if "sde_logprob" in result.output:  # FlowGRPO: capture the PPO log-prob slice
                rec.update(sde_logprob=result.output["sde_logprob"],
                           prev=to_host(result.output["prev"]).copy(),
                           sample=to_host(result.output["latents"]).copy(),
                           sigma_t=st.sigmas[i],
                           sigma_next=st.sigmas[i + 1])
            st.trajectory.append(rec)
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["video"]},
                          metrics={"denoise_steps": float(st.step_idx)},
                          behavior=st.trajectory or None)
