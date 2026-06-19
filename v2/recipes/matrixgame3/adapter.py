"""Matrix-Game 3.0 torch adapter (GPU backend) — declared on the card via ``ComponentSpec.adapter``
so the MG3 recipe is self-contained (no edit to the shared ``_make_dit`` dispatch in
``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only on a GPU box.

``MatrixGame3DiT`` is the autoregressive world-model DiT (``MatrixGame3WanModel``, 30 layers, dim 3072,
in/out_channels=48, patch (1,2,2), ``use_memory=True``, ``sigma_theta=0.8``). It departs from ``WanDiT``
in three load-bearing ways:

  1. PER-TOKEN timestep, NOT ``sigma*1000`` scalar. The model's ``timestep`` arg is a flattened
     ``[1, T*(H/2)*(W/2)]`` LongTensor whose value is the raw FlowUniPC scheduler timestep, with the
     ``cond_frames`` rows zeroed (those latent frames are pinned to the conditioning image and must not be
     denoised). When ``x_memory`` is present, an all-zeros ``timestep_memory`` block is concatenated in
     front. The loop builds the *scalar* timestep + ``cond_frames`` + per-clip conditioning bundle and
     hands them through ``cond=``; this adapter expands them to the per-token tensor INTERNALLY so the
     loop's dit-call stays ``dit(latent, text_embed, sigma)`` (ToyDiT-compatible — the cosmos2 pattern).
  2. Action / camera / memory conditioning: ``mouse_cond`` / ``keyboard_cond`` / ``x_memory`` /
     ``timestep_memory`` / ``mouse_cond_memory`` / ``keyboard_cond_memory`` / ``c2ws_plucker_emb`` /
     ``memory_latent_idx`` / ``predict_latent_idx`` are passed straight through. The loop assembles all of
     them (numpy host work — Plücker / extrinsics / memory-index selection) and stuffs them in ``cond``.
  3. The forward returns a velocity (``flow_prediction``) — a bare tensor ``[B,48,T,H,W]`` — which the
     FlowUniPC scheduler (driven by the loop) converts velocity->x0 internally in ``step()``.

The forward runs inside ``set_forward_context(current_timestep=<scalar t>, attn_metadata=None)`` (the
FastVideo attention layers read ``attn_metadata`` via the forward context). The 5B DiT is kept eager and
resident (no offload group). BRINGUP: written-not-run; the per-token timestep packing, the
``predict_latent_idx`` ``(start,end)`` tuple (the model expands it to a device ``arange``), and the
memory/plucker tensor dtypes must be GPU-verified against a real ``Matrix-Game-3.0`` checkpoint.
"""
from __future__ import annotations

from typing import Any

import torch

from v2.platform.backends.torch_backend import TorchComponent


class MatrixGame3DiT(TorchComponent):
    """``dit(latent[C=48,T,H,W], text_embed[seq,dim], sigma, cond=<MG3 bundle>) -> velocity[C,T,H,W]``.

    ``cond`` is the per-step conditioning bundle the ``MatrixGame3DenoiseLoop`` assembles (a plain dict).
    Keys (all optional except ``cond_frames``):
      * ``cond_frames`` (int)      — leading latent frames pinned to the conditioning image (1 first clip,
                                     4 later); their per-token timestep rows are zeroed.
      * ``mouse_cond`` / ``keyboard_cond`` (numpy [1, frames, dim]) — per-clip action streams.
      * ``x_memory`` / ``timestep_memory`` / ``mouse_cond_memory`` / ``keyboard_cond_memory`` (numpy) —
        autoregressive KV-memory tensors (None on the first clip / degenerate single-clip path).
      * ``c2ws_plucker_emb`` (numpy) — camera Plücker embeddings (memory blocks concatenated in front).
      * ``memory_latent_idx`` (list[int] | None) — history indices the memory tensors index into.
      * ``predict_latent_idx`` (tuple (start, end)) — the clip's latent-frame span (the model expands it
        to a device ``arange``); REQUIRED, the model uses it for rotary positions.
    """

    def __init__(self, module: Any, *, device: Any, dtype: Any) -> None:
        super().__init__(module, device=device, dtype=dtype)
        # patch_size[1:] -> (H/2, W/2) token folding for the per-token timestep packing. (1,2,2) default.
        ps = getattr(module, "patch_size", (1, 2, 2))
        self.patch_t, self.patch_h, self.patch_w = int(ps[0]), int(ps[1]), int(ps[2])

    def _per_token_timestep(self, hs: torch.Tensor, sigma_value: float, cond_frames: int,
                            x_memory: torch.Tensor | None) -> torch.Tensor:
        """Build the flattened per-token timestep [1, T*(H/p)*(W/p)]. The raw FlowUniPC timestep value fills
        every token; ``cond_frames`` latent rows are zeroed (pinned, not denoised); an all-zeros memory block
        is prepended when ``x_memory`` is present."""
        _b, _c, t, h, w = hs.shape
        tokens_per_frame = (h // self.patch_h) * (w // self.patch_w)
        tt = hs.new_full((t, tokens_per_frame), float(sigma_value))
        if cond_frames > 0:
            tt[:cond_frames].zero_()
        tt = tt.flatten().unsqueeze(0)
        if x_memory is not None:
            mem_tokens = x_memory.shape[2] * (x_memory.shape[3] // self.patch_h) * (x_memory.shape[4] // self.patch_w)
            tt = torch.cat([tt.new_zeros((1, mem_tokens)), tt], dim=1)
        return tt

    @torch.no_grad()
    def __call__(self,
                 latent: Any,
                 text_embed: Any,
                 sigma: float,
                 context: Any = None,
                 *,
                 cond: dict[str, Any] | None = None) -> Any:
        hs = self._t(latent)  # [1, 48, T, H, W]
        ehs = self._t(text_embed)
        cond = cond or {}
        cond_frames = int(cond.get("cond_frames", 0))
        x_memory = self._t(cond.get("x_memory")) if cond.get("x_memory") is not None else None
        # FlowUniPC timestep value: the loop passes the *raw* scheduler timestep (~sigma*1000), already
        # resolved, as ``sigma`` here (NOT a normalized 0..1 sigma). See the loop for the convention.
        timestep = self._per_token_timestep(hs, float(sigma), cond_frames, x_memory)

        def _opt(key: str) -> Any:
            v = cond.get(key)
            return self._t(v) if v is not None else None

        with self._ctx(current_timestep=float(sigma)):
            velocity = self.module(
                hs,
                ehs,
                timestep,
                mouse_cond=_opt("mouse_cond"),
                keyboard_cond=_opt("keyboard_cond"),
                x_memory=x_memory,
                timestep_memory=_opt("timestep_memory"),
                mouse_cond_memory=_opt("mouse_cond_memory"),
                keyboard_cond_memory=_opt("keyboard_cond_memory"),
                c2ws_plucker_emb=_opt("c2ws_plucker_emb"),
                memory_latent_idx=cond.get("memory_latent_idx"),
                predict_latent_idx=cond.get("predict_latent_idx"),
            )
        return self._n(velocity)  # rectified-flow velocity (FlowUniPC.step does velocity->x0)
