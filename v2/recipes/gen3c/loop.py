"""Gen3CDenoiseLoop — a TRUE EDM (EDMEulerScheduler) denoiser with camera/3D-cache conditioning hooks.

GEN3C is Cosmos-Predict2's EDM denoiser extended with camera-controlled video conditioning. Unlike the
Cosmos recipe (which folds EDM preconditioning into a *Karras* schedule), GEN3C's pipeline FORCES a
diffusers ``EDMEulerScheduler(sigma_max=80, sigma_min=0.0002, sigma_data=0.5)`` at runtime
(``Gen3CPipeline.initialize_pipeline``), so this loop reproduces that scheduler's exact σ schedule and
step math (see ``build_edm_euler_sigmas`` in ``sampler.py``):

  * σ schedule: ``EDMEulerScheduler``'s defaults are ``sigma_schedule='karras'`` (ρ=7) and
    ``final_sigmas_type='zero'`` — a Karras ρ ramp ``σ_max → σ_min`` then a single appended terminal
    ``0.0`` (verified bit-identical to the diffusers scheduler); the loop integrates the ``num_steps+1``
    points pairwise. Latent init = ``randn·√(σ_max²+σ_data²)`` (``init_noise_sigma``).
  * per step ``i`` (σ = ``sigmas[i]``, gamma=0 since s_churn=0 → ``σ_hat = σ``):
      - frame-replace BEFORE the forward (``_augment_noise_with_latent``): augment the conditioning
        latent with ``N(0,1)·σ_cond``, EDM-precondition, reverse ``c_in`` to put it back in the
        unscaled latent frame, and write it into ``x`` at ``cond_indicator`` positions;
      - ``model_input = x·c_in`` (``scale_model_input``); ``c_in = 1/√(σ²+σ_data²)``;
      - DiT(model_input, text, σ, cond=<pose bundle>) → raw output; the adapter embeds the model
        timestep ``c_noise = 0.25·log σ`` and concats the conditioning kwargs INTERNALLY (82 ch);
      - CFG (when enabled): cond branch with the REAL pose buffer; uncond branch with a ZEROED pose
        buffer + neg text; ``pred = cond + scale·(cond − uncond)``. NOTE default guidance_scale=1.0 +
        ``cfg_behavior='legacy'`` ⇒ NO uncond branch by default;
      - ``x0 = c_skip·x + c_out·pred`` (``precondition_outputs``); ``c_skip = σ_data²/(σ²+σ_data²)``,
        ``c_out = σ·σ_data/√(σ²+σ_data²)``;
      - frame-replace on the OUTPUT: ``x0 = ind·cond_latent + (1−ind)·x0`` (anchors the first frame);
      - Euler: ``v = (x − x0)/σ``; ``x_next = x + (σ_next − σ)·v`` (the platform ``FLOW_MATCH_STEP``,
        identical to ``EDMEulerScheduler.step``'s ODE-derivative update).

Camera / MoGe-depth / 3D-cache conditioning fills ``condition_video_pose`` / ``conditioning_latents`` /
``cond_indicator`` from program slots; ALL are ``None`` for the registered degenerate **t2v** path → the
frame-replace branches are inert and the DiT's internal pose concat is zeroed, matching how the fastvideo
pipeline degrades when no image/camera is given. Faithful port of ``Gen3CDenoisingStage.forward`` +
``Gen3CLatentPreparationStage`` (fastvideo/pipelines/stages/gen3c_stages.py).

The DiT call is ``dit(model_input, text_embed, sigma, cond=bundle)`` — ``ToyDiT`` accepts and ignores
``cond``, so the loop CPU-verifies on the toy backend.
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
from v2.recipes.gen3c.sampler import build_edm_euler_sigmas, edm_init_sigma
from v2.recipes.wan21.loop import latent_shape

GEN3C_LATENT_CHANNELS = 16  # config.in_channels (VAE latent); the 1+pose+1 channels are added in the DiT
GEN3C_TEMPORAL_RATIO = 8  # AutoencoderKLGen3CTokenizer target temporal compression (121 -> 16 latent frames)
GEN3C_SPATIAL_RATIO = 8


class Gen3CDenoiseLoop:

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 precision,
                 expert,
                 cost,
                 sigma_max: float = 80.0,
                 sigma_min: float = 0.0002,
                 sigma_data: float = 0.5,
                 sigma_conditional: float = 0.001,
                 latent_channels: int = GEN3C_LATENT_CHANNELS,
                 spatial_ratio: int = GEN3C_SPATIAL_RATIO,
                 temporal_ratio: int = GEN3C_TEMPORAL_RATIO):
        self.loop_id = loop_id
        self.cfg = cfg  # carried for the WorkPlan op-structure key; x0-space CFG is done here
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.sigma_conditional = sigma_conditional
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        sig = build_edm_euler_sigmas(req.diffusion.num_steps, sigma_max=self.sigma_max, sigma_min=self.sigma_min)
        shape = latent_shape(req,
                             model,
                             channels=self.latent_channels,
                             spatial_ratio=self.spatial_ratio,
                             temporal_ratio=self.temporal_ratio)
        # EDM latent init: randn · init_noise_sigma where init_noise_sigma = sqrt(σ_max² + σ_data²).
        x = (rng.standard_normal(shape) * edm_init_sigma(self.sigma_max, self.sigma_data)).astype("float32")
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"video": x},
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) for s in sig])  # the adapter derives c_noise = 0.25·log σ from σ
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        # Camera / 3D-cache conditioning written by the conditioning program node. ALL None for the
        # degenerate t2v path → the frame-replace branches below are inert and the DiT's pose concat is
        # zeroed (matching the fastvideo pipeline when no image/camera is given).
        st.scratch["conditioning_latents"] = ctx.slots.get("conditioning_latents")
        st.scratch["cond_indicator"] = ctx.slots.get("cond_indicator")
        st.scratch["uncond_indicator"] = ctx.slots.get("uncond_indicator")
        st.scratch["condition_video_pose"] = ctx.slots.get("condition_video_pose")
        st.scratch["condition_video_input_mask"] = ctx.slots.get("condition_video_input_mask")
        # Pixel H,W for the DiT's padding mask (resized internally). Carried so the adapter builds it.
        st.scratch["pixel_height"] = int(req.diffusion.height)
        st.scratch["pixel_width"] = int(req.diffusion.width)
        st.plugin_state["cfg"] = {}
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        t = st.timesteps[i]  # the raw σ; the adapter maps σ → c_noise = 0.25·log σ
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
        aug = float(self.sigma_conditional)
        pose = st.scratch.get("condition_video_pose")
        input_mask = st.scratch.get("condition_video_input_mask")
        pix_h, pix_w = st.scratch.get("pixel_height"), st.scratch.get("pixel_width")

        def _bundle(pose_buf: Any) -> dict | None:
            # The GEN3C conditioning bundle the adapter concats internally (None for pure t2v -> the DiT
            # builds zeros). ``pose_buf`` is zeroed on the CFG uncond branch.
            if pose is None and input_mask is None:
                return None
            return {
                "condition_video_pose": pose_buf,
                "condition_video_input_mask": input_mask,
                "condition_video_augment_sigma": aug,
                "height": pix_h,
                "width": pix_w,
            }

        def _x0(dit: Any, latent_in: np.ndarray, text_embed: Any, ind: Any, pose_buf: Any) -> np.ndarray:
            # EDM preconditioning + (gated) frame-replace conditioning, returning the x0 prediction for one
            # CFG branch. ``ind`` is the (un)cond indicator; None for t2v -> no frame injection.
            s = float(sigma_t)
            c_in = 1.0 / (s**2 + sd**2)**0.5
            c_skip = sd**2 / (s**2 + sd**2)
            c_out = s * sd / (s**2 + sd**2)**0.5
            # active_indicator is zeroed once σ drops to/below the conditioning-augment σ (the fastvideo
            # ``_augment_noise_with_latent`` guard: ``aug >= σ`` -> stop replacing).
            cur_ind = (ind * 0 if (ind is not None and aug >= s) else ind)
            lat = np.array(latent_in, dtype="float32")
            if cur_ind is not None and cond_lat is not None:  # frame-replace on the INPUT (inert for t2v)
                c_in_aug = 1.0 / (aug**2 + sd**2)**0.5
                cn = st.rng.standard_normal(lat.shape).astype("float32")
                # augment_latent = (cond + noise·aug); precondition (·c_in_aug); reverse the step's c_in.
                cf = (cond_lat + cn * aug) * c_in_aug / c_in
                lat = cur_ind * cf + (1 - cur_ind) * lat
            model_input = (lat * c_in).astype("float32")  # scale_model_input
            np_pred = np.asarray(dit(model_input, text_embed, s, cond=_bundle(pose_buf)), dtype="float32")
            x0 = c_skip * x + c_out * np_pred  # precondition_outputs (x_t is the UNSCALED x)
            if cur_ind is not None and cond_lat is not None:  # frame-replace on the OUTPUT (anchor frame-0)
                x0 = cur_ind * cond_lat + (1 - cur_ind) * x0
            return x0

        def run(model, override=None):
            dit = model.component(expert_id)
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                cond_x0 = _x0(dit, x, pe, cond_ind, pose)
                if do_cfg:
                    # CFG zeros the POSE buffer (not just text) on the uncond branch.
                    uncond_pose = (np.zeros_like(pose) if pose is not None else None)
                    uncond_x0 = _x0(dit, x, ne, uncond_ind, uncond_pose)
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
                        label=f"gen3c.denoise.{i}",
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
