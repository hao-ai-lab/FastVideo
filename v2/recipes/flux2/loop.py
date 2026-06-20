"""Flux2DenoiseLoop — FLUX.2 dual-stream MMDiT denoise on the BFL empirical-mu flow-match schedule.

FLUX.2 is a flow-match (velocity-prediction) image model, but it is NOT a Wan-shaped port (see
``v2/recipes/flux2/card.py`` for the component-level deltas). The deltas that force a NEW loop, all faithful
to ``fastvideo/pipelines/basic/flux_2`` + ``fastvideo/pipelines/stages/denoising.py``:

  * **Schedule** — the σ grid is the BFL empirical-mu grid (``flux2_sigmas`` here: ``linspace(1, 1/N, N)``
    shifted by ``compute_empirical_mu(image_seq_len, N)`` via the FlowMatchEuler exponential time-shift),
    NOT ``FlowShiftPolicy.build_schedule``'s resolution-bucket shift. ``image_seq_len`` is the PACKED token
    count ``latent_h * latent_w``.
  * **Packed latent geometry** — the DiT operates on ``num_channels_latents=64`` channels at HALF spatial
    (``latent_h = (H//8)//2``); the VAE decode unpacks the 2×2 patches back to 16ch full-spatial. So the
    loop initializes a ``(64, T=1, (H//8)//2, (W//8)//2)`` latent (image: ``T==1``).
  * **Embedded guidance, single forward** — FLUX.2-dev folds a guidance scalar into the DiT forward
    (``EmbeddedGuidance`` policy: one conditioned branch, no uncond), and FLUX.2-klein uses no guidance at
    all. There is NO classic 2-branch CFG, so the loop runs a single ``dit(x, pe, σ)`` call per step.
  * **img_ids / txt_ids RoPE + the ×1000 timestep/guidance split** — these are GPU-path concerns the
    ``Flux2DiT`` torch adapter handles INTERNALLY (it derives img_ids/txt_ids from the latent shape +
    text-embed length, passes σ DIRECTLY as the timestep, and the embedded guidance raw — the DiT
    multiplies both by 1000 itself). The loop therefore calls the dit exactly like the CPU ``ToyDiT``
    (``dit(latent, text_embed, sigma)``), so it CPU-verifies unchanged.

The velocity is integrated with the shared deterministic flow-match Euler step
(``x_next = x + (σ_next − σ)·v`` via ``FLOW_MATCH_STEP``) — identical solver to Wan/LTX-2.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2.core.enums import ExecutionProfile, WorkUnitKind
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
from v2.platform.backends.toy import LATENT_CHANNELS
from v2.recipes.flux2.sampler import flux2_sigmas

# Real FLUX.2 packed geometry: the VAE has latent_channels=16 at 8× spatial compression; the DiT sees
# the 2×2-packed latent -> 16·4 = 64 channels at HALF spatial. (config in_channels = num_channels_latents
# = 64; the VAE's _unpatchify_latents does C//4 -> 16ch full-spatial on decode.)
FLUX2_PACKED_CHANNELS = 64
FLUX2_SPATIAL_RATIO = 8  # VAE spatial_compression_ratio
FLUX2_PACK = 2  # 2×2 patch packing -> half spatial


def flux2_latent_geometry(req, model=None) -> tuple[tuple[int, int, int, int], int]:
    """Packed latent shape + packed image_seq_len. On the GPU backend the DiT/VAE require the true FLUX.2
    geometry; the CPU toy uses a tiny deterministic stand-in (so the loop control flow is exercised with
    real numbers). Returns ``((C, T, latent_h, latent_w), image_seq_len)`` where ``image_seq_len`` is the
    packed token count ``latent_h*latent_w`` the empirical-mu schedule keys on. Image model: ``T == 1``."""
    d = req.diffusion
    if model is not None and getattr(getattr(model, "platform", None), "device", "cpu") == "cuda":
        latent_h = max(1, (d.height // FLUX2_SPATIAL_RATIO) // FLUX2_PACK)
        latent_w = max(1, (d.width // FLUX2_SPATIAL_RATIO) // FLUX2_PACK)
        return (FLUX2_PACKED_CHANNELS, 1, latent_h, latent_w), latent_h * latent_w
    latent_h = max(2, d.height // 240)
    latent_w = max(2, d.width // 240)
    return (LATENT_CHANNELS, 1, latent_h, latent_w), latent_h * latent_w


class Flux2DenoiseLoop:
    """N flow-match Euler steps over the full packed latent, single conditioned forward per step
    (embedded guidance for dev, no guidance for klein). Mirrors ``WanDenoiseLoop`` structure; the FLUX.2
    schedule + packed geometry are the only deltas. All per-request state lives in ``LoopState``
    (interleave-safe)."""

    def __init__(self, *, loop_id, cfg, precision, expert, packed_channels: int = FLUX2_PACKED_CHANNELS):
        self.loop_id = loop_id
        self.cfg = cfg  # EmbeddedGuidance (dev) — carried for the WorkPlan op-structure key
        self.precision = precision
        self.expert = expert
        self.packed_channels = packed_channels

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        shape, image_seq_len = flux2_latent_geometry(req, model)
        sig = flux2_sigmas(req.diffusion.num_steps, image_seq_len)
        # FlowMatchEuler init_noise_sigma == 1.0 for this scheduler, so x starts at plain randn.
        x = rng.standard_normal(shape).astype("float32")
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile,
                       rng=rng,
                       seed=seed,
                       latents={"image": x},
                       sigmas=[float(s) for s in sig],
                       timesteps=[float(s) * 1000.0 for s in sig])  # bookkeeping; the adapter re-derives
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        # embedded-guidance scalar (dev=4.0; klein=None -> no guidance). The Flux2DiT adapter reads it and
        # folds it into the forward; there is no uncond branch.
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        st.scratch["stream_image"] = bool(req.outputs.stream.get("image"))
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
        branches = self.cfg.branches_this_step(sctx, cfg_state)  # ["cond"] for EmbeddedGuidance
        x = st.latents["image"]
        pe = st.cond["prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        cfg, precision = self.cfg, self.precision

        def _velocity(model: Any, x_: Any, sigma_t_: float, pe_: Any, scale_: float) -> np.ndarray:
            # Single conditioned forward; EmbeddedGuidance.combine is the identity on the cond branch. The
            # DiT call is toy-compatible (latent, text_embed, sigma) — img_ids/txt_ids/guidance/×1000 are
            # built INSIDE the Flux2DiT adapter, so the same loop drives both the CPU toy and the GPU DiT.
            dit = model.component(expert_id)
            preds = {b: dit(x_, pe_, sigma_t_) for b in branches}
            return precision.cast(cfg.combine(preds, scale_, sctx, cfg_state))

        def run(model, override=None):
            kernels = model.platform.kernels
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                velocity = _velocity(model, x, sigma_t, pe, scale)
            x_next = kernels.get(FLOW_MATCH_STEP)(precision.cast(x), velocity, sigma_t, sigma_next)
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": x_next.astype("float32")
            })

        def graph_fn(model, ws):
            # Capturable op-structure: reads every per-step input from the static workspace (never closure).
            st_t, st_n = float(ws["sigma_t"]), float(ws["sigma_next"])
            velocity = _velocity(model, ws["x"], st_t, ws["pe"], float(ws["scale"]))
            x_next = model.platform.kernels.get(FLOW_MATCH_STEP)(precision.cast(ws["x"]), velocity, st_t, st_n)
            np.copyto(ws["out"], x_next.astype("float32"))
            return StepResult(output={
                "noise_pred": np.asarray(velocity, dtype="float32"),
                "latents": np.array(ws["out"], copy=True)
            })

        cond_bytes = int(np.asarray(pe).nbytes) if pe is not None else 0
        res = ResourceRequest(
                              resident_bytes=int(x.nbytes) + cond_bytes,
                              peak_activation_bytes=int(x.nbytes))
        emits = []
        if st.scratch.get("stream_image"):
            from v2.core.request.streams import StreamChunk
            emits.append(StreamChunk(stream_id=st.request_id, modality="image", seq=i, data=x, preview=True))
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.DIFFUSION_STEP,
                        shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                                 dims=tuple(x.shape),
                                                 dtype=precision.compute_dtype,
                                                 extra=(("cfg", type(cfg).__name__), )),
                        resources=res,
                        payload={
                            "branch": "embedded",
                            "step": i
                        },
                        run=run,
                        label=f"flux2.denoise.{i}",
                        emits=emits,
                        capturable=True,
                        graph_key=(tuple(sorted(branches)), expert_id, precision.scheduler_step_in_fp32),
                        graph_fn=graph_fn,
                        graph_inputs={
                            "x": x,
                            "sigma_t": sigma_t,
                            "sigma_next": sigma_next,
                            "pe": pe,
                            "scale": scale
                        })

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["image"] = result.output["latents"]
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({
                "step": st.step_idx,
                "sigma": st.sigmas[st.step_idx],
                "velocity": np.asarray(result.output["noise_pred"]).copy(),
                "latents": np.asarray(st.latents["image"]).copy(),
            })
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        return LoopResult(outputs={"latents": st.latents["image"]},
                          metrics={"denoise_steps": float(st.step_idx)},
                          behavior=st.trajectory or None)
