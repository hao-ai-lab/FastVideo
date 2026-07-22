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
    latent: Any    # [1, C, T, h, w], transformer dtype
    timestep: Any  # [1] float tensor, 0..1000 scale (sigma * 1000)
    text: Any      # [1, 512, 4096] UMT5 embeddings

    def forward(self, transformer: Any) -> Any:
        """The one place the diffusers calling convention lives."""
        return transformer(hidden_states=self.latent, timestep=self.timestep,
                           encoder_hidden_states=self.text, return_dict=False)[0]


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
        # Compute dtype comes from the card, never from parameter introspection
        # (the DiT is legitimately mixed-dtype: fp32 islands inside bf16).
        from fastvideo2.loading import declared_torch_dtype
        st.scratch["dit_dtype"] = declared_torch_dtype(instance.card.components["transformer"])
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
            x_in = x.to(dit_dtype)
            with torch.no_grad():
                v_cond = WanForwardInputs(x_in, t, text).forward(dit).to(torch.float32)
                if guidance == 1.0:
                    v = v_cond
                else:
                    v_neg = WanForwardInputs(x_in, t, neg).forward(dit).to(torch.float32)
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
