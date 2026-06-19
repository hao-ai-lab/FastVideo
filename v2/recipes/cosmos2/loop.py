"""CosmosDenoiseLoop — EDM-Karras preconditioning folded into a flow-match Euler integrator.

Cosmos-Predict2 is not a plain flow-match model despite the pipeline wrapping a
``FlowMatchEulerDiscreteScheduler``. The network is an EDM denoiser ``F_θ`` whose preconditioned output
reconstructs ``x0`` (``x0 = c_skip·x + c_out·F_θ(x·c_in)``, ``sigma_data=1``); CFG combines in x0 space;
then ``x0`` is converted to a flow-match velocity ``(x - x0)/σ`` for the Euler update
``x_next = x + (σ_next - σ)·v``. The σ schedule is Karras (ρ=7, σ_max=80 -> σ_min=0.002), not the
flow-shift linspace, and the latent starts at ``randn·σ_max``. Port of
``fastvideo/pipelines/stages/denoising.py:CosmosDenoisingStage`` (see that file for the exact math).

video2world (frame_replace) conditioning is threaded (``conditioning_latents``/``cond_indicator`` from
slots) but is ``None`` for the registered t2v preset -> the gated injection is inert; the loop degrades
to pure t2v exactly as the fastvideo pipeline does when no image/video is given.
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
from v2.loop.sampler import build_karras_sigmas
from v2.platform import FLOW_MATCH_STEP
from v2.recipes.wan21.loop import latent_shape

COSMOS_LATENT_CHANNELS = 16  # config.in_channels(17) - 1 (the condition_mask channel)
COSMOS_TEMPORAL_RATIO = 4
COSMOS_SPATIAL_RATIO = 8


class CosmosDenoiseLoop:

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 precision,
                 expert,
                 cost,
                 sigma_max: float = 80.0,
                 sigma_min: float = 0.002,
                 sigma_data: float = 1.0,
                 rho: float = 7.0,
                 augment_sigma: float = 0.001,
                 latent_channels: int = COSMOS_LATENT_CHANNELS,
                 spatial_ratio: int = COSMOS_SPATIAL_RATIO,
                 temporal_ratio: int = COSMOS_TEMPORAL_RATIO):
        self.loop_id = loop_id
        self.cfg = cfg  # carried for the WorkPlan op-structure key; x0-space CFG is done here
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.rho = rho
        self.augment_sigma = augment_sigma
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        sig = build_karras_sigmas(req.diffusion.num_steps,
                                  sigma_max=self.sigma_max,
                                  sigma_min=self.sigma_min,
                                  rho=self.rho)
        shape = latent_shape(req,
                             model,
                             channels=self.latent_channels,
                             spatial_ratio=self.spatial_ratio,
                             temporal_ratio=self.temporal_ratio)
        x = (rng.standard_normal(shape) * float(self.sigma_max)).astype("float32")  # EDM: randn·σ_max
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"video": x},
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) * 1000.0 for s in sig])  # model timestep = σ·1000 (large for EDM)
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        # video2world (frame_replace): VAE-encoded conditioning latents + cond/uncond indicators. None for
        # the t2v preset -> the gated injection below is skipped (pure t2v, matching the fastvideo pipeline).
        st.scratch["conditioning_latents"] = ctx.slots.get("conditioning_latents")
        st.scratch["cond_indicator"] = ctx.slots.get("cond_indicator")
        st.scratch["uncond_indicator"] = ctx.slots.get("uncond_indicator")
        st.plugin_state["cfg"] = {}
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        t = st.timesteps[i]  # σ·1000, the model timestep
        expert_id = self.expert.expert_for(StepContext(i, t, sigma_t))
        x = st.latents["video"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        precision = self.precision
        sd = float(self.sigma_data)
        do_cfg = scale != 1.0 and ne is not None
        cond_lat = st.scratch.get("conditioning_latents")
        cond_ind = st.scratch.get("cond_indicator")
        uncond_ind = st.scratch.get("uncond_indicator")
        aug = float(self.augment_sigma)

        def _x0(model: Any, dit: Any, latent_in: np.ndarray, text_embed: Any, ind: Any) -> np.ndarray:
            # EDM preconditioning + (gated) frame-replace conditioning, returning the x0 prediction for one
            # CFG branch. ``ind`` is the (un)cond indicator; None for t2v -> no frame injection.
            s = float(sigma_t)
            c_in = 1.0 / (s**2 + sd**2)**0.5
            c_skip = sd**2 / (s**2 + sd**2)
            c_out = s * sd / (s**2 + sd**2)**0.5
            cur_ind = (ind * 0 if (ind is not None and aug >= s) else ind)
            lat = np.array(latent_in, dtype="float32")
            if cur_ind is not None and cond_lat is not None:  # video2world frame injection (inert for t2v)
                c_in_aug = 1.0 / (aug**2 + sd**2)**0.5
                cn = st.rng.standard_normal(lat.shape).astype("float32")
                cf = (cond_lat + cn * aug) * c_in_aug / c_in
                lat = cur_ind * cf + (1 - cur_ind) * lat
            model_input = (lat * c_in).astype("float32")
            np_pred = np.asarray(dit(model_input, text_embed, t), dtype="float32")
            x0 = c_skip * x + c_out * np_pred
            if cur_ind is not None and cond_lat is not None:
                x0 = cur_ind * cond_lat + (1 - cur_ind) * x0
            return x0

        def run(model, override=None):
            dit = model.component(expert_id)
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                cond_x0 = _x0(model, dit, x, pe, cond_ind)
                if do_cfg:
                    uncond_x0 = _x0(model, dit, x, ne, uncond_ind)
                    final_x0 = cond_x0 + scale * (cond_x0 - uncond_x0)
                else:
                    final_x0 = cond_x0
                velocity = precision.cast((x - final_x0) / max(float(sigma_t), 1e-6))
            x_next = model.platform.kernels.get(FLOW_MATCH_STEP)(precision.cast(x), velocity, sigma_t, sigma_next)
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": x_next.astype("float32")
            })

        cond_bytes = sum(int(np.asarray(e).nbytes) for e in (pe, ne) if e is not None)
        res = ResourceRequest(compute_seconds=self.cost.predict(int(np.prod(x.shape)), 2.0 if do_cfg else 1.0),
                              resident_bytes=int(x.nbytes) + cond_bytes,
                              peak_activation_bytes=int(x.nbytes))
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.DIFFUSION_STEP,
                        shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                                 dims=tuple(x.shape),
                                                 dtype=precision.compute_dtype,
                                                 extra=(("cfg", type(self.cfg).__name__), ("edm", True))),
                        resources=res,
                        payload={
                            "branch": "edm",
                            "step": i
                        },
                        run=run,
                        label=f"cosmos.denoise.{i}",
                        capturable=False)  # EDM x0-space CFG + (gated) host-RNG frame injection -> eager path

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
