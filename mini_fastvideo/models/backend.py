"""Toy numpy components — the CPU-testable backend (design_v3 §17 honesty note).

There is no GPU/torch/weights in this environment, so the heavy Wan/LTX neural forwards are
represented by small, deterministic numpy components. They exercise the *real* loop control
flow, CFG/flow-shift policies, scheduler steps, cache reuse, parity gates, and training-method
math with real numbers — just with a toy network instead of a 1.3B DiT.

The contract these toys honor is what matters:
  * deterministic given (weights, inputs) → bit-reproducible (the interleave/parity gates rely on this);
  * same prompt → same text embedding → feature-cache reuse is correct;
  * a velocity ("flow") prediction the flow-match sampler consumes (Wan/LTX prediction_type).

On a GPU box, ``ComponentSpec.factory`` swaps these for lazy torch adapters wrapping the real
``fastvideo.models`` modules + weights (see each model's ``components.py``); the loops, policies,
scheduler, caches, parity, and training code are unchanged. That is the whole point of the
(recipe, runtime) separation.
"""
from __future__ import annotations

import hashlib

import numpy as np

LATENT_CHANNELS = 4
TEXT_SEQ = 4
TEXT_DIM = 8


def _seed_from(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


class ToyTextEncoder:
    """Deterministic text→embedding (same text ⇒ same embedding ⇒ feature-cache reuse works)."""

    def __init__(self, seq: int = TEXT_SEQ, dim: int = TEXT_DIM):
        self.seq, self.dim = seq, dim

    def encode(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(_seed_from("txt:" + (text or "<empty>")))
        return (rng.standard_normal((self.seq, self.dim)) * 0.1).astype("float32")


class ToyDiT:
    """A tiny deterministic velocity predictor (stands in for the 1.3B DiT).

    velocity = tanh( Wx·latent  +  Wt·σ  +  s·mean(text_embed)  +  c·mean(context) )
    Channel-mixing via ``Wx`` makes the trajectory non-trivial; ``context`` carries causal
    chunk history for the self-forcing chunk_rollout loop.
    """

    def __init__(self, channels: int = LATENT_CHANNELS, seed: int = 0):
        self.C = channels
        rng = np.random.default_rng(seed)
        self.w_x = (rng.standard_normal((channels, channels)) * 0.15).astype("float32")
        self.w_t = (rng.standard_normal(channels) * 0.1).astype("float32")
        self.s_text = 0.5
        self.s_ctx = 0.3

    def _pre_tanh(self, latent: np.ndarray, text_embed, sigma: float, context) -> np.ndarray:
        mixed = np.tensordot(self.w_x, latent, axes=([1], [0]))          # [C,...] channel mix
        mixed = mixed + self.w_t.reshape((self.C,) + (1,) * (latent.ndim - 1)) * float(sigma)
        cond = float(np.mean(text_embed)) if text_embed is not None else 0.0
        mixed = mixed + self.s_text * cond
        if context is not None:
            mixed = mixed + self.s_ctx * float(np.mean(context))
        return mixed

    def __call__(self, latent: np.ndarray, text_embed: np.ndarray | None, sigma: float,
                 context: np.ndarray | None = None) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float32)
        return np.tanh(self._pre_tanh(latent, text_embed, sigma, context)).astype("float32")

    # --- minimal trainable surface (so training methods do real optimizer steps) --------- #
    def clone(self) -> "ToyDiT":
        c = ToyDiT.__new__(ToyDiT)
        c.C, c.s_text, c.s_ctx = self.C, self.s_text, self.s_ctx
        c.w_x, c.w_t = self.w_x.copy(), self.w_t.copy()
        return c

    def blend_from(self, other: "ToyDiT", decay: float) -> None:
        """EMA / decay-blended-old-policy update: self ← decay·self + (1-decay)·other (design_v3 §10)."""
        self.w_x = (decay * self.w_x + (1.0 - decay) * other.w_x).astype("float32")
        self.w_t = (decay * self.w_t + (1.0 - decay) * other.w_t).astype("float32")

    def copy_from(self, other: "ToyDiT") -> None:
        self.w_x, self.w_t = other.w_x.copy(), other.w_t.copy()

    def mse_grad_step(self, latent: np.ndarray, text_embed, sigma: float, target: np.ndarray,
                      lr: float, context=None, weight: float = 1.0) -> tuple[float, float]:
        """One exact SGD step minimizing ``weight·MSE(forward(latent), target)`` w.r.t. ``w_x``.

        Returns (loss, grad_norm). The grad_norm is the #1396-style per-method regression signal.
        This is a *real* gradient (chain rule through tanh), so the toy student actually learns —
        the structural stand-in for the FSDP/optimizer step a GPU trainer would run."""
        latent = np.asarray(latent, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        z = self._pre_tanh(latent, text_embed, sigma, context)
        pred = np.tanh(z)
        err = pred - target
        loss = float(weight * np.mean(err ** 2))
        n = err.size
        g = weight * (2.0 * err / n) * (1.0 - pred ** 2)
        axes = (list(range(1, g.ndim)), list(range(1, latent.ndim)))
        grad_wx = np.tensordot(g, latent, axes=axes)                     # [C_out, C_in]
        grad_norm = float(np.linalg.norm(grad_wx))
        self.w_x = (self.w_x - lr * grad_wx).astype("float32")
        return loss, grad_norm


class ToyVAE:
    """Tiny deterministic VAE. encode: video→latent (mean-pool + channel proj); decode: latent→video."""

    def __init__(self, channels: int = LATENT_CHANNELS, spatial: int = 8, temporal: int = 4, seed: int = 1):
        self.C = channels
        self.spatial = spatial
        self.temporal = temporal
        rng = np.random.default_rng(seed)
        self.dec_proj = (rng.standard_normal((3, channels)) * 0.2).astype("float32")

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """latent [C,T,H,W] -> video [3, T, H*spatial, W*spatial] (deterministic upsample)."""
        latent = np.asarray(latent, dtype=np.float32)
        rgb = np.tensordot(self.dec_proj, latent, axes=([1], [0]))       # [3,T,H,W]
        up = np.repeat(np.repeat(rgb, self.spatial, axis=2), self.spatial, axis=3)
        return np.tanh(up).astype("float32")

    def encode(self, video: np.ndarray) -> np.ndarray:
        video = np.asarray(video, dtype=np.float32)
        pooled = video[:self.C]                                          # crude: take C channels
        return pooled.astype("float32")
