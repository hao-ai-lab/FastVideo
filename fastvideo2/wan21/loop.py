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


def dmd_sigma_table(shift: float, num_train_timesteps: int = 1000) -> list[tuple[float, float]]:
    """fastvideo's FlowMatchEulerDiscreteScheduler training table: (timestep,
    sigma) pairs where sigma is the shift-warped linspace and timestep is
    sigma*1000. Because timesteps ARE warped sigmas x1000, looking a DMD step
    like 757 up in this table returns sigma ~= 0.757 (nearest entry) almost
    independently of shift — shift only sets table density. We reproduce the
    lookup, not an approximation, for bit-faithfulness to the authority."""
    out = []
    for i in range(num_train_timesteps):
        s = (num_train_timesteps - i) / num_train_timesteps
        sigma = shift * s / (1.0 + (shift - 1.0) * s)
        out.append((sigma * num_train_timesteps, sigma))
    return out


def dmd_sigma_for(t: float, table: list[tuple[float, float]]) -> float:
    """argmin |table_timestep - t| — fastvideo's scheduler lookup, verbatim."""
    return min(table, key=lambda row: abs(row[0] - t))[1]


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


class WanDMDLoop(WanDenoiseLoop):
    """Few-step DMD student sampler (FastWan family): at each declared
    timestep, one CFG-free forward predicts velocity, the clean latent is
    recovered as x0 = x - sigma*v (in fp64 — fastvideo's DmdDenoisingStage
    convention), then re-noised to the next timestep with FRESH per-request
    noise: x = (1-sigma_next)*x0 + sigma_next*eps. The last step emits x0.

    The step count and timesteps are properties of the distilled weights —
    hence a distinct semantics id: a card whose provenance assumes this loop
    cannot validate against the base Euler sampler, and vice versa. Authority
    for these artifacts is fastvideo main (their training stack), so sigma
    lookup reproduces main's scheduler table exactly.
    """

    semantics = "wan.dmd.x0renoise/v1"

    def __init__(self, *, loop_id: str, timesteps: tuple[float, ...] = (1000.0, 757.0, 522.0),
                 shift: float = 8.0, **geometry: Any):
        super().__init__(loop_id=loop_id, **geometry)
        self.timesteps = tuple(float(t) for t in timesteps)
        self.shift = float(shift)

    def init(self, request: Any, instance: Any, inputs: Mapping[str, Any]) -> LoopState:
        import torch
        st = super().init(request, instance, inputs)
        table = dmd_sigma_table(self.shift)
        st.sigmas = [dmd_sigma_for(t, table) for t in self.timesteps]
        st.scratch["timesteps"] = list(self.timesteps)
        # Re-seed the initial latent for the DMD schedule (sigma_0 ~= 1.0 ->
        # plain randn); the SAME generator stream then supplies the re-noising
        # draws, mirroring main's per-request batch.generator usage.
        gen = torch.Generator(instance.device).manual_seed(request.seed)
        st.latents = torch.randn(st.latents.shape, generator=gen,
                                 device=instance.device, dtype=torch.float32)
        st.scratch["gen"] = gen
        return st

    def next(self, st: LoopState) -> "WorkPlan | Done":
        i = st.step_idx
        if i >= len(st.sigmas):
            return Done()
        sigma = st.sigmas[i]
        t_model = st.scratch["timesteps"][i]
        sigma_next = st.sigmas[i + 1] if i + 1 < len(st.sigmas) else None
        x, text = st.latents, st.cond["text"]
        instance = st.scratch["instance"]
        dit_dtype = st.scratch["dit_dtype"]
        gen = st.scratch["gen"]

        def run() -> dict:
            import torch
            dit = instance.component("transformer")
            t = torch.tensor([t_model], device=x.device, dtype=torch.float32)
            autocast = torch.amp.autocast("cuda", dtype=dit_dtype,
                                          enabled=x.is_cuda and dit_dtype != torch.float32)
            with torch.no_grad(), autocast:
                v = WanForwardInputs(x, t, text).forward(dit)
            # x0 in fp64, exactly like fastvideo's pred_noise_to_pred_video
            x0 = (x.double() - sigma * v.double())
            if sigma_next is None:
                return {"latents": x0.to(torch.float32)}
            noise = torch.randn(x.shape, generator=gen, device=x.device,
                                dtype=torch.float32)
            nxt = (1.0 - sigma_next) * x0 + sigma_next * noise.double()
            return {"latents": nxt.to(torch.float32)}

        return WorkPlan(label=f"{self.loop_id}.{i}", step=i, run=run,
                        meta={"sigma": sigma, "timestep": t_model})
