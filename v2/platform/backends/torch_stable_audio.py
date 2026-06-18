"""Stable Audio Open torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter``
so the recipe is self-contained (no edit to the shared ``_make_dit``/``_make_vae``/``_make_text_encoder``
dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only on a GPU box.

Stable Audio is an AUDIO model and differs from every existing (video) adapter on several axes, all of
which are absorbed HERE so ``StableAudioDenoiseLoop`` keeps the cosmos2-style ``dit(latent, text_embed,
sigma)`` call shape (and so the CPU toy backend works unchanged):

* ``StableAudioDiT`` — the v-prediction (EDM-v) network. forward is ``forward(x[B,64,L], t[B], *,
  cross_attn_cond[B,seq,768], global_embed[B,1536]) -> v[B,64,L]`` (1-D audio latent, NOT [C,T,H,W]).
  cross_attn_cond/global_embed are keyword-only and the DiT does NOT do CFG internally — the loop hands
  a single batch, CFG is combined in the loop (faithful to ``stable_audio/stages/denoising.py:_DiTAdapter``).
  The adapter packs the loop's flat ``text_embed`` payload into (cross_attn_cond, global_embed) and passes
  the RAW continuous sigma as the timestep ``t`` (NOT sigma*1000 (Wan), NOT 0..1 (LTX-2); k-diffusion /
  EDM c_noise convention — BRINGUP risk B). It returns the raw network output; the v->x0 (VDenoiser)
  preconditioning lives in the loop.
* ``OobleckVAE`` — 1-D conv audio VAE (weight_norm + Snake1d, hop_length=2048). encode(audio[2,N]) ->
  raw latent (``dist.sample()``, NO mean/std normalization — passthrough latent space, BRINGUP risk D);
  decode(latent[64,L]) -> waveform[2,M]. 1-D marshalling (no 5-D unsqueeze). Exposes ``hop_length`` and
  ``sampling_rate`` so the loop sizes the latent and the program slices [start,end] seconds.
* ``StableAudioConditioner`` — the SA multi-conditioner (T5-base @ max_length=128 + duration
  NumberConditioners). encode(prompt, seconds_start, seconds_total) -> packs the (cross_attn_cond,
  cross_attn_mask, global_embed) triple into a single numpy payload the loop stashes in cond. NOTE: the
  shared ``_MAKERS`` dispatch has no ``conditioner`` kind and ``TextEncoderLoader`` will not build a
  ``StableAudioMultiConditioner`` — so the card declares this component with ``kind="text_encoder"`` +
  this explicit adapter, and the adapter is constructed ``cls(module, tokenizer, ...)`` (the tokenizer
  extra the text-encoder maker passes is unused — the conditioner owns its own T5 tokenizer). The maker
  still calls ``TextEncoderLoader`` though, which loads a generic T5, NOT the SA conditioner — so wiring
  this on a GPU box needs the SA ``ConditionerLoader`` path (BRINGUP risk: the conditioner load).

All three are BRINGUP (written-not-run): there is no GPU/k_diffusion in this env. See the loop for the
dpmpp-3m-sde GPU path that drives ``StableAudioDiT`` through ``K.external.VDenoiser``.
"""
from __future__ import annotations

import numpy as np

from v2.platform.backends.torch_backend import TorchComponent, _to_numpy

# Packed conditioning payload layout. The loop treats the conditioner output as one flat "text_embed"
# numpy array; the DiT adapter splits it back into cross_attn_cond + global_embed by these widths so the
# loop->dit call stays ``dit(latent, text_embed, sigma)`` (cosmos2-shape) and the CPU toy still runs.
_GLOBAL_COND_DIM = 1536  # StableAudioArchConfig.global_cond_dim
_CROSS_COND_DIM = 768  # StableAudioArchConfig.cond_token_dim

# The global embed (1536) is laid down as ``_GLOBAL_HEAD_ROWS`` leading rows of width 768 (1536 = 2*768),
# then the cross-attn tokens (each 768-wide) follow. Lets the whole conditioning ride as ONE [N,768] array.
_GLOBAL_HEAD_ROWS = _GLOBAL_COND_DIM // _CROSS_COND_DIM  # = 2


def pack_conditioning(cross_attn_cond: np.ndarray, global_embed: np.ndarray) -> np.ndarray:
    """Pack (cross_attn_cond[seq,768], global_embed[1536]) -> a single [2+seq, 768] numpy payload: the
    first 2 rows are the 1536-d global embed reshaped to [2,768], rows 2.. are the cross-attn tokens. The
    loop carries this opaque array as ``text_embed``; ``StableAudioDiT.__call__`` (and the CPU toy, which
    just means-pools it) consume it. Keeps the loop modality-agnostic — it never sees the SA tuple."""
    g = np.asarray(global_embed, dtype="float32").reshape(-1)
    head = np.zeros((_GLOBAL_HEAD_ROWS, _CROSS_COND_DIM), dtype="float32")
    head.reshape(-1)[:min(g.shape[0], _GLOBAL_COND_DIM)] = g[:_GLOBAL_COND_DIM]  # row-major lay-down
    cross = np.asarray(cross_attn_cond, dtype="float32").reshape(-1, _CROSS_COND_DIM)
    return np.concatenate([head, cross], axis=0)  # [2 + seq, 768]; rows0-1=global, rows2..=cross-attn


def unpack_conditioning(payload: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``pack_conditioning``: -> (cross_attn_cond[seq,768], global_embed[1536])."""
    p = np.asarray(payload, dtype="float32")
    global_embed = p[:_GLOBAL_HEAD_ROWS].reshape(-1)[:_GLOBAL_COND_DIM].copy()
    cross = p[_GLOBAL_HEAD_ROWS:]
    return cross, global_embed


class StableAudioDiT(TorchComponent):
    """``dit(latent[64,L], text_embed=<packed cond>, sigma) -> raw v-prediction[64,L]``.

    ``sigma`` is the RAW continuous sigma (k-diffusion / EDM c_noise; ~0.3..500). The loop hands the
    ALREADY-VDenoiser-input-scaled latent (``x·c_in``) — this adapter is the bare network ``F_theta`` and
    returns the raw output; the v->x0 reconstruction + x0-space CFG live in ``StableAudioDenoiseLoop``
    (faithful to ``_DiTAdapter`` + ``K.external.VDenoiser``). Single batch; CFG batching is the loop's job.
    """

    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        import torch
        cross_np, global_np = unpack_conditioning(text_embed)
        hs = self._t(latent)  # [1, 64, L]
        cross = self._t(cross_np)  # [1, seq, 768]
        glob = self._t(global_np)  # [1, 1536]
        t = torch.tensor([float(sigma)], device=self.device, dtype=self.dtype)  # RAW sigma, NOT *1000
        with torch.no_grad(), self._ctx():
            out = self.module(hs, t, cross_attn_cond=cross, global_embed=glob)
        return self._n(out)  # RAW v-prediction (loop does VDenoiser v->x0)


class OobleckVAE(TorchComponent):
    """1-D audio VAE. decode(latent[64,L]) -> waveform[channels, samples]; encode(audio[ch,N]) -> raw
    latent (NO mean/std normalization — SA's DiT operates directly in the Oobleck latent space; BRINGUP
    risk D). Exposes ``hop_length`` / ``sampling_rate`` (read off the loaded module)."""

    def __init__(self, module, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)
        self.hop_length = int(getattr(module, "hop_length", 2048))
        self.sampling_rate = int(getattr(module, "sampling_rate", 44100))

    def decode(self, latent):
        import torch
        z = self._t(latent).to(self.dtype)  # [1, 64, L]
        with torch.no_grad(), self._ctx():
            out = self.module.decode(z)
        wav = out.sample if hasattr(out, "sample") else out  # [1, channels, samples]
        return _to_numpy(wav.squeeze(0))  # [channels, samples]

    def encode(self, audio):
        import torch
        x = self._t(audio).to(self.dtype)  # [1, channels, samples]
        with torch.no_grad(), self._ctx():
            dist = self.module.encode(x)
        z = dist.sample() if hasattr(dist, "sample") else dist  # raw latent, NO normalization
        return _to_numpy(z.squeeze(0))  # [64, L]


class StableAudioConditioner(TorchComponent):
    """SA multi-conditioner (T5-base @ max_length=128 + duration NumberConditioners). ``encode`` packs the
    (cross_attn_cond, cross_attn_mask, global_embed) triple into the loop's flat payload via
    ``pack_conditioning``. The DiT consumes a single-batch payload; CFG (null/zero negative cross-attn) is
    built in the loop.

    BRINGUP: the shared text-encoder maker calls ``TextEncoderLoader`` (a generic T5), not the SA
    ``ConditionerLoader`` that builds ``StableAudioMultiConditioner`` — so on a GPU box this adapter must
    instead be handed a real ``StableAudioMultiConditioner`` module (e.g. via a recipe-local loader
    override / a ``conditioner`` component kind). The ``tokenizer`` extra the text-encoder maker passes is
    accepted and ignored (the conditioner owns its own T5 tokenizer)."""

    def __init__(self, module, tokenizer=None, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)
        # The active sub-conditioner ids (3 for SA-1.0: prompt+seconds_start+seconds_total; 2 for
        # SA-small: prompt+seconds_total). Read off the module so the variant is self-describing.
        self.cross_attention_cond_ids = tuple(
            getattr(module, "cross_attention_cond_ids", ("prompt", "seconds_start", "seconds_total")))

    def encode(self, prompt: str, seconds_start: float = 0.0, seconds_total: float = 10.0) -> np.ndarray:
        import torch
        all_vals = {
            "prompt": prompt or "",
            "seconds_start": float(seconds_start),
            "seconds_total": float(seconds_total)
        }
        cond_meta = [{k: all_vals[k] for k in self.cross_attention_cond_ids if k in all_vals}]
        with torch.no_grad(), self._ctx():
            cond = self.module(cond_meta, self.device)
            cross, _mask, glob = self.module.get_conditioning_inputs(cond)
        return pack_conditioning(_to_numpy(cross.squeeze(0)), _to_numpy(glob.squeeze(0)))
