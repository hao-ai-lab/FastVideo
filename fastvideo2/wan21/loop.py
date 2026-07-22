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
