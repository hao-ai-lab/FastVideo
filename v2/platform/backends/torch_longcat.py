"""LongCat-Video torch adapter (GPU backend) — declared on the card via ``ComponentSpec.adapter`` so
the LongCat recipe is self-contained (no edit to the shared ``_make_dit`` dispatch in ``torch_backend.py``).
Imported lazily by ``_explicit_adapter`` only on a GPU box.

* ``LongCatDiT`` — the flow-match velocity predictor. Mirrors ``WanDiT`` (``timestep = sigma*1000``,
  scalar ``[B]`` timestep that the DiT internally expands ``[B] -> [B, T_latent]`` for the per-latent-frame
  AdaLN time embedding) but with the critical LongCat sign convention: the fastvideo
  ``LongCatDenoisingStage`` does ``noise_pred = -noise_pred`` *before* the flow-match scheduler step. The v2
  loop integrates ``x + (sigma_next-sigma)*velocity`` with the velocity this adapter returns, so to reproduce
  that step exactly this adapter returns ``-velocity``. Forgetting the negation silently diverges.

LongCat reuses the existing ``WanVAE`` torch adapter unchanged (same ``AutoencoderKLWan`` mean/std
normalization) and the existing ``T5Encoder`` (UMT5, zero-padded to max_length=512 for the CFG concat
uniform-seq contract). Both are wired on the card by ``load_id`` alone — no adapter override needed.

The DiT runs its AdaLN modulation / residual gating + final projection under fp32 and casts the output to
float32; we keep the module at its native bf16 dtype (``_native_dtype`` in ``_make_dit``, matching the
fastvideo stage's hardcoded bf16 target) and let ``_n`` marshal the fp32 output back to numpy.

BRINGUP: written-not-run on CPU (no torch/weights here). GPU-verify against a real
``FastVideo/LongCat-Video-T2V-Diffusers`` checkpoint — confirm the bf16 dtype, the scalar-timestep
expansion, and ``encoder_attention_mask=None`` (the CaptionEmbedder simply skips the zeroing branch for
T2V, which is faithful since the v2 ``T5Encoder`` already zero-pads the embedding rows past real tokens).
"""
from __future__ import annotations

import torch

from v2.platform.backends.torch_backend import NUM_TRAIN_TIMESTEPS, TorchComponent


class LongCatDiT(TorchComponent):
    """``dit(latent[C,T,H,W], text_embed[seq,4096], sigma) -> -velocity[C,T,H,W]``.

    Real forward (``fastvideo/models/dits/longcat.py``):
    ``forward(hidden_states[B,C=16,T,H,W], encoder_hidden_states[B,N_text,4096], timestep[B]|[B,T],
    encoder_attention_mask=None, ...) -> velocity[B,C_out=16,T,H,W]`` (cast to float32).

    The loop hands the raw ``sigma`` (1->0); this adapter forms ``timestep = sigma*1000`` (the Wan/diffusers
    convention LongCat shares) as a scalar ``[B]`` tensor — the DiT's ``forward`` expands ``[B] -> [B, T_latent]``
    internally for the AdaLN time embedding. It returns the negated velocity to fold in the fastvideo stage's
    ``noise_pred = -noise_pred``."""

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        hs = self._t(latent)  # [1, C=16, T, H, W]
        ehs = self._t(text_embed)  # [1, N_text, 4096]
        b = hs.shape[0]
        # timestep = sigma*1000 (BRINGUP risk B). Scalar [B]; the DiT expands [B] -> [B, T_latent] internally
        # for the per-latent-frame AdaLN time embedding. (I2V/VC would build an explicit [B, T_latent] with
        # cond frames zeroed — deferred, T2V uses the scalar path.)
        ts = float(sigma) * NUM_TRAIN_TIMESTEPS
        timestep = torch.full((b, ), ts, device=self.device, dtype=self.dtype)
        with self._ctx(current_timestep=ts):
            velocity = self.module(hidden_states=hs,
                                   encoder_hidden_states=ehs,
                                   timestep=timestep,
                                   encoder_attention_mask=None)
        # CRITICAL: negate so the loop's FLOW_MATCH_STEP reproduces the fastvideo stage's
        # negate-then-scheduler-step. The DiT already casts to float32; _n marshals fp32 -> numpy.
        return self._n(-velocity)
