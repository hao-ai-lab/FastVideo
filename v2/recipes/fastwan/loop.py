"""FastWanDMDLoop — Distribution-Matching-Distillation few-step denoise for FastWan.

FastWan is a DMD-distilled Wan: instead of N flow-match Euler steps it runs a handful (3) of
*predict-x0-then-renoise* steps over a fixed list of discrete model timesteps. This is NOT a plain
flow-match Euler integrator (``WanDenoiseLoop``), so it ships its own ``next``/``advance`` and uses the
shared policies as a library.

Faithful port of ``fastvideo/pipelines/stages/denoising.py:DmdDenoisingStage`` (+ ``pred_noise_to_pred_video``
in ``fastvideo/models/utils.py``). The exact math, per discrete model timestep ``t_i`` (e.g. [1000, 757, 522]):

  * the DMD scheduler is ``FlowMatchEulerDiscreteScheduler(shift=8.0)`` built over the full 1000 train
    timesteps (the stage hardcodes shift=8.0 regardless of the pipeline ``flow_shift``); a discrete
    timestep ``t`` maps to a (shifted) sigma via ``s = t/1000``, ``sigma = shift·s / (1 + (shift-1)·s)``.
  * single forward (DMD is a distilled one-branch model: guidance is *embedded* via the ``guidance``
    kwarg, NOT classifier-free) -> ``pred_noise``.
  * x0 reconstruction (flow prediction): ``pred_video = x - sigma_t · pred_noise``.
  * re-noise to the next discrete timestep: ``x = (1 - sigma_next)·pred_video + sigma_next·noise``
    (the scheduler's ``add_noise``), with fresh Gaussian noise; the final step keeps ``pred_video``.

The init latent is ``randn · sigma(t0)``; with t0=1000 -> s=1.0 -> sigma=1.0, so unit-variance noise.

The dit call is kept toy-compatible (``dit(latent, text_embed, sigma, cond=...)``) so the CPU ToyDiT
drives the loop unchanged. The re-noise host-RNG branch makes every step eager (capturable=False), like
the cosmos2 EDM loop. i2v conditioning (``i2v_cond`` first-frame [mask|cond]) is threaded for the TI2V
variant but is ``None`` for the registered t2v preset -> the dit forward is the plain path (degrades to
t2v exactly as the fastvideo pipeline does when no image is given).
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
from v2.recipes.wan21.loop import latent_shape

# DMD distilled-scheduler constants (DmdDenoisingStage hardcodes FlowMatchEulerDiscreteScheduler(shift=8.0)
# over the default 1000 train timesteps).
DMD_SHIFT = 8.0
DMD_NUM_TRAIN_TIMESTEPS = 1000

# Real Wan2.2-TI2V-5B VAE (AutoencoderKLWan) compression: z_dim=48, 16x spatial, 4x temporal — the
# loadable FullAttn target's geometry. The FastWan2.1 variants use 16/8/4 (Wan2.1 defaults).
FASTWAN_TI2V_LATENT_CHANNELS = 48
FASTWAN_TI2V_SPATIAL_RATIO = 16
FASTWAN_TI2V_TEMPORAL_RATIO = 4


def _shifted_sigma(timestep: float, shift: float = DMD_SHIFT, num_train: int = DMD_NUM_TRAIN_TIMESTEPS) -> float:
    """Map a discrete model timestep -> the DMD scheduler's (shifted) flow sigma.

    Mirrors ``FlowMatchEulerDiscreteScheduler``'s constructor: ``s = t/num_train``, then the flow-shift
    ``sigma = shift·s / (1 + (shift-1)·s)``. The DMD stage looks the sigma up from the scheduler's sigma
    array by nearest-timestep match; since the scheduler is the full 1000-step grid (``sigma_i = t_i/1000``
    shifted) and the DMD timesteps are integers in [0, 1000], the closed form reproduces that lookup
    exactly (the nearest grid timestep is the timestep itself)."""
    s = float(timestep) / float(num_train)
    return shift * s / (1.0 + (shift - 1.0) * s)


class FastWanDMDLoop:
    """DMD few-step loop: a fixed list of discrete timesteps, predict-x0-then-renoise each step."""

    def __init__(self,
                 *,
                 loop_id: str,
                 cfg: Any,
                 precision: Any,
                 expert: Any,
                 cost: Any,
                 denoising_steps: list[int],
                 shift: float = DMD_SHIFT,
                 latent_channels: int = FASTWAN_TI2V_LATENT_CHANNELS,
                 spatial_ratio: int = FASTWAN_TI2V_SPATIAL_RATIO,
                 temporal_ratio: int = FASTWAN_TI2V_TEMPORAL_RATIO) -> None:
        self.loop_id = loop_id
        self.cfg = cfg  # carried for the WorkPlan op-structure key; DMD is single-branch
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.denoising_steps = [int(t) for t in denoising_steps]
        self.shift = shift
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        # The DMD schedule is the model's fixed denoising_step_list (NOT req.num_steps): the distilled
        # checkpoint was trained for these exact timesteps. We honor a request override only if it matches
        # the list length, else we use the model list verbatim (faithful to the fastvideo stage, which
        # reads pipeline_config.dmd_denoising_steps and ignores num_inference_steps for the schedule).
        steps = list(self.denoising_steps)
        sigmas = [_shifted_sigma(t, self.shift) for t in steps]
        shape = latent_shape(req,
                             model,
                             channels=self.latent_channels,
                             spatial_ratio=self.spatial_ratio,
                             temporal_ratio=self.temporal_ratio)
        x = (rng.standard_normal(shape) * float(sigmas[0])).astype("float32")  # randn · sigma(t0) (=1.0)
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"video": x},
                       sigmas=[float(s) for s in sigmas],
                       timesteps=[float(t) for t in steps])  # DMD model timesteps ARE the integers
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        # i2v (TI2V variant): the program writes the CLIP image embeds + the first-frame [mask|cond] latent
        # into slots; absent for the registered t2v preset -> None -> the dit forward is the plain path.
        st.scratch["i2v_cond"] = ctx.slots.get("i2v_cond")
        st.scratch["i2v_img_embeds"] = ctx.slots.get("i2v_img_embeds")
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        n = len(st.timesteps)
        if i >= n:
            return Done()
        sigma_t = st.sigmas[i]
        timestep = st.timesteps[i]
        expert_id = self.expert.expert_for(StepContext(i, timestep, sigma_t))
        x = st.latents["video"]
        pe = st.cond["prompt_embeds"]
        precision = self.precision
        rng = st.rng
        is_last = i == n - 1
        sigma_next = 0.0 if is_last else st.sigmas[i + 1]
        i2v_ctx, i2v_cond = st.scratch.get("i2v_img_embeds"), st.scratch.get("i2v_cond")

        def _pred_noise(model: Any) -> np.ndarray:
            # Single distilled forward; DMD bakes guidance into the model (embedded), no CFG combine.
            # i2v_ctx/i2v_cond are None for t2v (plain forward); for i2v the Wan adapter concats the
            # conditioning latent and passes the CLIP embeds as encoder_hidden_states_image.
            dit = model.component(expert_id)
            return np.asarray(dit(x, pe, sigma_t, context=i2v_ctx, cond=i2v_cond), dtype="float32")

        def run(model, override=None):
            if override is not None and "noise_pred" in override:
                pred_noise = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                pred_noise = precision.cast(_pred_noise(model))
            # x0 reconstruction (flow prediction): pred_video = x - sigma_t · pred_noise.
            pred_video = precision.cast(np.asarray(x, dtype="float32") - sigma_t * pred_noise)
            if is_last:
                x_next = pred_video
            else:
                # Re-noise to the next discrete timestep: (1 - sigma_next)·x0 + sigma_next·noise.
                noise = rng.standard_normal(x.shape).astype("float32")
                x_next = precision.cast((1.0 - sigma_next) * pred_video + sigma_next * noise)
            return StepResult(output={
                "noise_pred": np.asarray(pred_noise, dtype="float32"),
                "latents": np.asarray(x_next, dtype="float32")
            })

        cond_bytes = int(np.asarray(pe).nbytes) if pe is not None else 0
        res = ResourceRequest(
            compute_seconds=self.cost.predict(int(np.prod(x.shape)), 1.0),  # single forward per step
            resident_bytes=int(x.nbytes) + cond_bytes,
            peak_activation_bytes=int(x.nbytes))
        from v2.request.streams import StreamChunk
        emits = []
        if st.scratch.get("stream_video"):
            emits.append(StreamChunk(stream_id=st.request_id, modality="video", seq=i, data=x, preview=True))
        return WorkPlan(
            loop_id=self.loop_id,
            instance_id=st.instance_id,
            kind=WorkUnitKind.DIFFUSION_STEP,
            shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                     dims=tuple(x.shape),
                                     dtype=precision.compute_dtype,
                                     extra=(("cfg", type(self.cfg).__name__), ("dmd", True))),
            resources=res,
            payload={
                "branch": "dmd",
                "step": i
            },
            run=run,
            label=f"fastwan.dmd.{i}",
            emits=emits,
            # The re-noise host-RNG branch (and the i2v conditioning thread) make the step non-capturable,
            # like the cosmos2 EDM loop -> the runtime eager-breaks it.
            capturable=False)

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
