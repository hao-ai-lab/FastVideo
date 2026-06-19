"""StableAudioDenoiseLoop — a v-prediction (EDM-v / VDenoiser) DPM++ denoise loop for Stable Audio Open.

Stable Audio is not a flow-match model and not an EDM-Karras model: it is a v-prediction network sampled
by k-diffusion's ``dpmpp-3m-sde`` (a 3rd-order multistep stochastic DPM-Solver++) over a polyexponential
sigma schedule (sigma_min=0.3, sigma_max=500, rho=1.0). So neither ``WanDenoiseLoop`` (flow-match) nor
``CosmosDenoiseLoop`` (EDM-Karras x0) can be reused — this needs a new loop.

Faithfulness vs ``fastvideo/pipelines/basic/stable_audio/stages/denoising.py``:
  * The schedule is polyexponential (``sampler.build_polyexponential_sigmas``), seeded noise scaled by
    sigmas[0], exactly like ``StableAudioLatentPreparationStage`` + ``StableAudioDenoisingStage``.
  * The network is a v-predictor wrapped by ``K.external.VDenoiser``: ``denoised = c_skip·x + c_out·F(c_in·x, c_noise)``
    with the EDM-v coefficients (sigma_data=1). This loop reproduces those coefficients exactly
    (``sampler.vdenoiser_x0``) so the x0 it integrates matches k-diffusion's VDenoiser on the GPU path.
  * CFG: ``denoised = uncond + (cond - uncond)·scale``, combined in x0 space (the
    ``_DiTAdapter`` convention; the negative cross-attn is null/zero, applied by the conditioning node).

GPU PATH (BRINGUP — k_diffusion is an external dep, not installed in this env): the faithful, lowest-risk
GPU route is to run the whole k-diffusion sampler inside one thunk rather than decomposing
``dpmpp-3m-sde`` into per-step v2 WorkPlans: ``K.sampling.sample_dpmpp_3m_sde(K.external.VDenoiser(adapter),
x, sigmas, ...)`` — a direct lift of the fastvideo stage. That needs a per-loop "single-shot" execution
mode the v2 ``ctx.execute`` per-step contract does not yet expose (BRINGUP: needs a loop-as-one-thunk hook).
For CPU-verification + per-step serving granularity, this loop ships a deterministic DPM-Solver++(2M)
recurrence in numpy over the same VDenoiser x0 — a faithful v-prediction multistep integrator that runs
end-to-end on the CPU toy. The stochastic higher-order ``dpmpp-3m-sde`` terms are the GPU-parity refinement
(BRINGUP) layered on top of this deterministic spine.

A2A (init_audio -> lowered sigma_max) and RePaint inpainting (per-step blend callback) are threaded as
inert hooks (None for the registered T2A preset) and documented BRINGUP — base port is pure text->audio.
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
from v2.recipes.stable_audio.sampler import (
    build_polyexponential_sigmas,
    dpmpp_2m_step,
    vdenoiser_x0,
)

# Oobleck VAE geometry: 64 io_channels; latent_len = sample_size // hop_length (hop_length=2048).
SA_IO_CHANNELS = 64
SA_HOP_LENGTH = 2048
SA_SAMPLE_SIZE = 2097152  # SA-1.0 fixed training window (~47.55s @ 44.1kHz); SA-small is 524288.


def audio_latent_len(req, model, *, sample_size: int, hop_length: int = SA_HOP_LENGTH) -> int:
    """Latent length the DiT/VAE operate over. The model ALWAYS samples a fixed-size latent
    (``sample_size // hop_length``) and the decode is sliced to [start,end] seconds afterwards — the
    duration knob does NOT shrink the latent (faithful to the fastvideo stage). The CPU toy uses a tiny
    deterministic stand-in so the loop math runs without the 1024-long real latent."""
    if model is not None and getattr(getattr(model, "platform", None), "device", "cpu") == "cuda":
        return max(1, sample_size // hop_length)
    return 8  # toy: short latent, enough to exercise the multistep recurrence


class StableAudioDenoiseLoop:

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 precision,
                 expert,
                 cost,
                 sigma_min: float = 0.3,
                 sigma_max: float = 500.0,
                 rho: float = 1.0,
                 io_channels: int = SA_IO_CHANNELS,
                 hop_length: int = SA_HOP_LENGTH,
                 sample_size: int = SA_SAMPLE_SIZE):
        self.loop_id = loop_id
        self.cfg = cfg  # carried for the WorkPlan op-structure key; x0-space CFG is done here
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.io_channels = io_channels
        self.hop_length = hop_length
        self.sample_size = sample_size

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        # Polyexponential schedule sigma_max -> sigma_min (NOT flow-shift, NOT Karras-rho-7).
        sig = build_polyexponential_sigmas(req.diffusion.num_steps,
                                           sigma_min=self.sigma_min,
                                           sigma_max=self.sigma_max,
                                           rho=self.rho)
        latent_len = audio_latent_len(req, model, sample_size=self.sample_size, hop_length=self.hop_length)
        shape = (self.io_channels, latent_len)  # 1-D audio latent [64, L]
        x = (rng.standard_normal(shape) * float(sig[0])).astype("float32")  # noise·sigma_max
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"audio": x},
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) for s in sig])  # RAW sigma is the model timestep (NOT *1000)
        # The conditioning node packs the SA (cross_attn_cond, global_embed) triple into one numpy payload
        # (see torch_stable_audio.pack_conditioning); the negative payload is the CFG uncond branch.
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        # A2A / RePaint (BRINGUP): inert for the T2A preset (None) -> the gated branches below are skipped.
        st.scratch["init_latent"] = ctx.slots.get("init_latent")
        st.scratch["inpaint_ref"] = ctx.slots.get("inpaint_reference_latent")
        st.scratch["inpaint_mask"] = ctx.slots.get("inpaint_mask_latent")
        st.scratch["prev_x0"] = None  # DPM++(2M) multistep memory: the previous step's x0 estimate
        st.plugin_state["cfg"] = {}
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        t = st.timesteps[i]  # raw sigma
        expert_id = self.expert.expert_for(StepContext(i, t, sigma_t))
        x = st.latents["audio"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        precision = self.precision
        do_cfg = scale != 1.0 and ne is not None
        prev_x0 = st.scratch.get("prev_x0")

        def _x0(dit, text_embed) -> np.ndarray:
            # VDenoiser preconditioning: feed the network c_in·x at c_noise(sigma), get the raw v output,
            # reconstruct x0 = c_skip·x + c_out·v (EDM-v, sigma_data=1). Faithful to K.external.VDenoiser.
            return vdenoiser_x0(x, float(sigma_t),
                                lambda scaled, c_noise: np.asarray(dit(scaled, text_embed, c_noise), dtype="float32"))

        def run(model, override=None):
            dit = model.component(expert_id)
            if override is not None and "x0" in override:
                final_x0 = precision.cast(np.asarray(override["x0"], dtype="float32"))
            else:
                cond_x0 = _x0(dit, pe)
                if do_cfg:
                    uncond_x0 = _x0(dit, ne)
                    final_x0 = uncond_x0 + (cond_x0 - uncond_x0) * scale  # x0-space CFG (uncond + (cond-uncond)·s)
                else:
                    final_x0 = cond_x0
            final_x0 = precision.cast(final_x0)
            # Deterministic DPM-Solver++(2M) step over (x, x0, sigma_t -> sigma_next), using prev_x0 for the
            # 2nd-order correction (1st-order Euler on the first step). The stochastic dpmpp-3m-sde terms
            # are the GPU-parity refinement (BRINGUP); this deterministic spine runs on the CPU toy.
            x_next = dpmpp_2m_step(x, final_x0, sigma_t, sigma_next, prev_x0)
            return StepResult(output={"x0": np.asarray(final_x0, dtype="float32"), "latents": x_next.astype("float32")})

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
                                                 extra=(("cfg", type(self.cfg).__name__), ("vpred", True))),
                        resources=res,
                        payload={
                            "branch": "vpred",
                            "step": i
                        },
                        run=run,
                        label=f"stable_audio.denoise.{i}",
                        capturable=False)  # v->x0 + x0-space CFG + multistep host state -> eager path

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["audio"] = result.output["latents"]
        st.scratch["prev_x0"] = np.asarray(result.output["x0"], dtype="float32")  # DPM++(2M) memory
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({
                "step": st.step_idx,
                "sigma": st.sigmas[st.step_idx],
                "x0": np.asarray(result.output["x0"]).copy(),
                "latents": np.asarray(st.latents["audio"]).copy(),
            })
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["audio"]},
                          metrics={"denoise_steps": float(st.step_idx)},
                          behavior=st.trajectory or None)
