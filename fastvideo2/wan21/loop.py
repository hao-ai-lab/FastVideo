"""WanDenoiseLoop — the production denoise loop, driven by the runtime.

The math is deliberately identical to ``reference.py`` (CFG over a shifted
flow-match Euler schedule, latents held fp32, DiT called in its own dtype);
the T2 gate exists to keep the two in lockstep. What this file adds over the
reference is the *contract*: kernel-free planning, per-step identity, all
mutable state in ``LoopState``, and a typed forward-input dataclass so nothing
model-specific rides an untyped kwarg.

Module import stays torch-free; torch is touched only inside methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fastvideo2.loop import Done, LoopState, WorkPlan


def flow_sigmas(num_steps: int, shift: float) -> list[float]:
    """The shifted flow-match schedule: t_i = (N-i)/N warped by
    sigma = shift*t / (1 + (shift-1)*t). sigma_0 = 1, sigma_N = 0."""
    out = []
    for i in range(num_steps + 1):
        t = (num_steps - i) / num_steps
        out.append(shift * t / (1.0 + (shift - 1.0) * t))
    return out


@dataclass(frozen=True)
class WanForwardInputs:
    """Everything one Wan DiT forward consumes — typed, so a new conditioning
    channel is a new field the adapter must handle, never a silently-dropped
    kwarg."""
    latent: Any    # [1, C, T, h, w], fp32 (official convention: cast under autocast)
    timestep: Any  # [1] float tensor, 0..1000 scale (sigma * 1000)
    text: Any      # [1, 512, 4096] UMT5 embeddings (the zero-pad is load-bearing)

    def forward(self, transformer: Any) -> Any:
        """The one place the official calling convention lives: list-based,
        unbatched, with the max sequence length derived from the patch grid.
        Callers provide official's precision regime (autocast bf16 on CUDA;
        plain fp32 for the exact-math path)."""
        pt, ph, pw = transformer.patch_size
        _, _, f, h, w = self.latent.shape
        seq_len = (f // pt) * (h // ph) * (w // pw)
        out = transformer([self.latent[0]], t=self.timestep,
                          context=[self.text[0]], seq_len=seq_len)[0]
        return out[None]


class WanDenoiseLoop:
    """N flow-match Euler steps with classifier-free guidance over the full
    latent. ``guidance_scale == 1.0`` runs the single conditioned forward."""

    semantics = "wan.flow_euler.cfg/v1"

    def __init__(self, *, loop_id: str, latent_channels: int = 16,
                 spatial_ratio: int = 8, temporal_ratio: int = 4):
        self.loop_id = loop_id
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio

    # --------------------------------------------------------------------- #
    def init(self, request: Any, instance: Any, inputs: Mapping[str, Any]) -> LoopState:
        import torch
        shape = (1, self.latent_channels,
                 (request.num_frames - 1) // self.temporal_ratio + 1,
                 request.height // self.spatial_ratio,
                 request.width // self.spatial_ratio)
        gen = torch.Generator(instance.device).manual_seed(request.seed)
        x = torch.randn(shape, generator=gen, device=instance.device, dtype=torch.float32)
        st = LoopState(loop_id=self.loop_id, request_id=request.request_id,
                       latents=x, sigmas=flow_sigmas(request.num_steps, request.shift))
        st.cond["text"] = inputs["text_embeds"]
        st.cond["neg_text"] = inputs["neg_text_embeds"]
        st.scratch["guidance"] = float(request.guidance_scale)
        st.scratch["capture"] = bool(request.capture_trajectory)
        st.scratch["instance"] = instance
        # Compute (autocast) dtype comes from the card's provenance — the
        # precision the artifact assumes — never from parameter introspection.
        # Storage dtype (ComponentSpec.dtype) is a separate axis: official
        # stores the DiT fp32 and computes bf16 under autocast.
        from fastvideo2.loading import torch_dtype
        st.scratch["dit_dtype"] = torch_dtype(instance.card.provenance.precision)
        return st

    def next(self, st: LoopState) -> "WorkPlan | Done":
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()
        sigma, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        x, text, neg = st.latents, st.cond["text"], st.cond["neg_text"]
        guidance = st.scratch["guidance"]
        instance = st.scratch["instance"]

        dit_dtype = st.scratch["dit_dtype"]

        def run() -> dict:
            import torch
            dit = instance.component("transformer")
            t = torch.tensor([sigma * 1000.0], device=x.device, dtype=torch.float32)
            # Official precision regime: fp32 latents under autocast in the
            # card-declared compute dtype (the model's own fp32 islands —
            # rope, time embedding, modulation — are autocast-disabled inside).
            autocast = torch.amp.autocast("cuda", dtype=dit_dtype,
                                          enabled=x.is_cuda and dit_dtype != torch.float32)
            with torch.no_grad(), autocast:
                v_cond = WanForwardInputs(x, t, text).forward(dit).to(torch.float32)
                if guidance == 1.0:
                    v = v_cond
                else:
                    v_neg = WanForwardInputs(x, t, neg).forward(dit).to(torch.float32)
                    v = v_neg + guidance * (v_cond - v_neg)
            # Euler step in fp32: x' = x + (sigma_next - sigma) * v
            return {"latents": x + (sigma_next - sigma) * v}

        return WorkPlan(label=f"{self.loop_id}.{i}", step=i, run=run,
                        meta={"sigma": sigma})

    def advance(self, st: LoopState, result: dict) -> LoopState:
        st.latents = result["latents"]
        if st.scratch["capture"]:
            st.trajectory.append(st.latents.detach().to("cpu", copy=True))
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> dict:
        return {"latents": st.latents, "trajectory": st.trajectory,
                "steps": st.step_idx}




# --------------------------------------------------------------------------- #
# FastWan (DMD) — authority: fastvideo-main. Semantics vendored from
# DmdDenoisingStage + FlowMatchEulerDiscreteScheduler
# @ e3f47dc2de2d1fa0c68c5839a0a41ed25b04a953.
# --------------------------------------------------------------------------- #
def dmd_inference_table(shift: float = 8.0,
                        num_train_timesteps: int = 1000) -> tuple[list[float], list[float]]:
    """The sigma table main's DMD stage ACTUALLY looks sigmas up in.

    Subtle and load-bearing: main's DmdDenoisingStage.__init__ HARDCODES a
    fresh internal ``FlowMatchEulerDiscreteScheduler(shift=8.0)`` — it ignores
    both the pipeline's prepared scheduler (so TimestepPreparationStage's
    ``set_timesteps(n)`` never reaches these lookups) and the config's
    flow_shift. The lookups therefore run against the INIT-time 1000-entry
    shift-warped training table, where timesteps are the warped sigmas x1000 —
    an argmin for t=757 lands on the row whose warped sigma is ~0.757, nearly
    independent of shift. Table math mirrors main's init exactly: numpy fp32
    linspace, fp32 division and warp.

    Returns (timesteps, sigmas) as python floats of the fp32 table.
    """
    import numpy as np
    ts = np.linspace(1, num_train_timesteps, num_train_timesteps,
                     dtype=np.float32)[::-1].copy()
    sig = ts / np.float32(num_train_timesteps)
    sig = np.float32(shift) * sig / (1 + (np.float32(shift) - 1) * sig)
    timesteps = [float(s * num_train_timesteps) for s in sig]
    return timesteps, [float(s) for s in sig]


def dmd_sigma_for(t: float, timesteps: list[float], sigmas: list[float]) -> float:
    """main's scheduler lookup: argmin |table_timestep - t|."""
    idx = min(range(len(timesteps)), key=lambda i: abs(timesteps[i] - t))
    return sigmas[idx]


@dataclass(frozen=True)
class WanFVForwardInputs:
    """One forward of the fastvideo-main-vendored DiT (``model_fv``): batched
    diffusers-style tensors, latents in main's BTCHW state layout (permuted to
    BCFHW at the call), integer timesteps. Typed so nothing rides a kwarg."""
    latent_btchw: Any   # [1, T, C, h, w] — main's DMD state layout
    timestep: Any       # [1] int64 tensor (main feeds the raw 1000/757/522)
    text: Any           # [1, 512, 4096] UMT5 embeddings, fp32

    def forward(self, transformer: Any) -> Any:
        """Returns velocity in BTCHW. Caller provides main's precision regime
        (autocast bf16, input pre-cast to the compute dtype)."""
        out = transformer(self.latent_btchw.permute(0, 2, 1, 3, 4),
                          self.text, self.timestep)
        return out.permute(0, 2, 1, 3, 4)


class WanDMDLoop(WanDenoiseLoop):
    """FastWan's 3-step DMD sampler, bit-faithful to fastvideo-main's
    DmdDenoisingStage: state kept in main's BTCHW layout; ONE CPU generator
    per request supplies the initial fp32 latents and then the bf16 renoise
    draws (stream order is part of the contract); x0 = x - sigma*v in fp64,
    stored back at the transformer's output dtype; renoise
    (1-sigma')*x0 + sigma'*noise with fp32 sigma tensors, result cast to the
    noise dtype. Sigmas come from ``dmd_inference_table`` — see its docstring
    for why that table is NOT the training table.
    """

    semantics = "wan.dmd.fvmain/v1"

    def __init__(self, *, loop_id: str, timesteps: tuple[int, ...] = (1000, 757, 522),
                 shift: float = 8.0, **geometry: Any):
        super().__init__(loop_id=loop_id, **geometry)
        self.timesteps = tuple(int(t) for t in timesteps)
        # NOTE: main hardcodes the DMD stage's internal scheduler at shift 8.0
        # regardless of config flow_shift; cards declare it explicitly.
        self.shift = float(shift)

    def init(self, request: Any, instance: Any, inputs: Mapping[str, Any]) -> LoopState:
        import torch
        st = LoopState(loop_id=self.loop_id, request_id=request.request_id)
        table_t, table_s = dmd_inference_table(self.shift)
        st.sigmas = [dmd_sigma_for(t, table_t, table_s) for t in self.timesteps]
        # main: batch.generator = [torch.Generator("cpu").manual_seed(seed)];
        # initial latents drawn on CPU in BTCHW fp32 (dtype follows the fp32
        # prompt embeds), then moved to device.
        gen = torch.Generator("cpu").manual_seed(request.seed)
        shape = (1,
                 (request.num_frames - 1) // self.temporal_ratio + 1,
                 self.latent_channels,
                 request.height // self.spatial_ratio,
                 request.width // self.spatial_ratio)
        st.latents = torch.randn(shape, generator=gen, dtype=torch.float32
                                 ).to(instance.device)
        st.cond["text"] = inputs["text_embeds"]
        st.scratch["gen"] = gen
        st.scratch["capture"] = bool(request.capture_trajectory)
        st.scratch["instance"] = instance
        from fastvideo2.loading import torch_dtype
        st.scratch["dit_dtype"] = torch_dtype(instance.card.provenance.precision)
        return st

    def next(self, st: LoopState) -> "WorkPlan | Done":
        i = st.step_idx
        if i >= len(self.timesteps):
            return Done()
        t_int = self.timesteps[i]
        sigma = st.sigmas[i]
        sigma_next = st.sigmas[i + 1] if i + 1 < len(self.timesteps) else None
        x, text = st.latents, st.cond["text"]
        instance = st.scratch["instance"]
        dit_dtype = st.scratch["dit_dtype"]
        gen = st.scratch["gen"]

        def run() -> dict:
            import torch
            dit = instance.component("transformer")
            device = x.device
            t = torch.tensor([t_int], dtype=torch.int64, device=device)
            autocast = torch.amp.autocast("cuda", dtype=dit_dtype,
                                          enabled=x.is_cuda and dit_dtype != torch.float32)
            with torch.no_grad(), autocast:
                v = WanFVForwardInputs(x.to(dit_dtype), t, text).forward(dit)
            # pred_noise_to_pred_video: fp64 math, cast back to the PRED dtype
            sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device).double()
            x0 = (x.double() - sigma_t * v.double()).to(v.dtype)
            if sigma_next is None:
                return {"latents": x0}
            # renoise draw: CPU generator, BTCHW, in x0's (bf16) dtype
            noise = torch.randn(x.shape, dtype=x0.dtype, generator=gen).to(device)
            # add_noise: fp32 sigma TENSOR (promotes), result cast to noise dtype
            s_next = torch.tensor(sigma_next, dtype=torch.float32, device=device)
            nxt = ((1 - s_next) * x0 + s_next * noise).type_as(noise)
            return {"latents": nxt}

        return WorkPlan(label=f"{self.loop_id}.{i}", step=i, run=run,
                        meta={"sigma": sigma, "timestep": t_int})

    def finalize(self, st: LoopState) -> dict:
        # hand back the engine's BCTHW convention (main permutes at stage end)
        latents = st.latents.permute(0, 2, 1, 3, 4)
        return {"latents": latents, "trajectory": st.trajectory, "steps": st.step_idx}
