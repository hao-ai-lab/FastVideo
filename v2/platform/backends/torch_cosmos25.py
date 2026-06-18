"""Cosmos-Predict2.5 torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter``
so the Cosmos2.5 recipe is self-contained (no edit to the shared ``_make_dit``/``_make_vae``/
``_make_text_encoder`` dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only
on a GPU box.

The three Cosmos2.5 deltas vs Wan (all GPU-path, all carried HERE so the loop stays toy-compatible):

* ``Cosmos25DiT`` — a flow-match velocity DiT, but the per-step call differs from ``WanDiT`` in three
  load-bearing ways (see ``fastvideo/pipelines/stages/denoising.py:Cosmos25DenoisingStage``):
    1. timestep is a PER-FRAME 2D tensor ``[B, T]`` filled with the PLAIN sigma (in [0,1]) — NOT Wan's
       scalar ``sigma*1000``. (The fastvideo stage feeds ``t*0.001`` of the 0..1000 schedule, i.e. the
       shifted sigma directly; ``Cosmos25DenoiseLoop`` already hands us that plain sigma.)
    2. the forward REQUIRES a ``condition_mask[B,1,T,H,W]`` (zeros for t2v) + a ``padding_mask[B,1,H,W]``
       (ones for t2v) every step — the model concats them internally (16 -> 17 -> 18ch patch_embed),
       so we feed the RAW 16-channel latent (the OPPOSITE of Wan i2v, where the adapter pre-concats).
    3. an ``fps`` scalar is required for RoPE temporal scaling (NTK fps-modulation).
  The model output is the rectified-flow velocity (a bare tensor); the loop integrates it with the
  shared ``FLOW_MATCH_STEP`` Euler solver and combines branches with the standard ``ClassicCFG``.
* ``Cosmos25WanVAE`` — a Wan-style causal 3D VAE (z=16, 8x/4x), BUT unlike ``AutoencoderKLWan`` its
  ``encode``/``decode`` already apply the (z-mean)/std normalization INTERNALLY (the Cosmos latent
  contract: encode -> normalized, decode <- normalized). So this is a thin marshalling wrapper (like
  ``LTX2VAE``) that must NOT re-normalize — reusing the v2 ``WanVAE`` adapter would double-normalize.
* ``Cosmos25Reason1Encoder`` — Qwen2.5-VL multimodal LM via ``compute_text_embeddings_online``: per-layer
  mean-normalized hidden states concatenated along the feature dim -> a 100352-dim (= num_hidden_layers *
  hidden_size = 28*3584) sequence the DiT's internal 100352->1024 GELU crossattn projection consumes.
  Faithful to ``Cosmos25TextEncodingStage`` (calls ``compute_text_embeddings_online`` rather than
  re-implementing the normalization). BRINGUP: heavy (a 7B LM + 100352-dim embeds); GPU-only.
"""
from __future__ import annotations

import torch

from v2.platform.backends.torch_backend import TorchComponent, _to_numpy

# Cosmos2.5 t2v feeds the DiT the PLAIN shifted sigma in [0,1] as the per-frame timestep — NOT
# ``sigma * num_train_timesteps`` (the Wan convention would corrupt the output). See blocker (1).
COSMOS25_DEFAULT_FPS = 24  # RoPE temporal scaling; the registered t2v preset's default fps.


class Cosmos25DiT(TorchComponent):
    """``dit(latent[C,T,h,w], text_embed[seq,100352], sigma) -> velocity[C,T,h,w]``.

    The loop hands the RAW 16-channel latent and the PLAIN sigma (in [0,1]); this adapter builds the
    mandatory per-frame timestep + zero ``condition_mask`` + ones ``padding_mask`` + ``fps`` the Cosmos2.5
    forward requires (the model concats the masks itself -> 18ch patch_embed input). Returns the bare
    velocity tensor; ``Cosmos25DenoiseLoop`` does the ClassicCFG combine + the flow-match Euler step."""

    def __init__(self, module, *, device, dtype, fps: int = COSMOS25_DEFAULT_FPS):
        super().__init__(module, device=device, dtype=dtype)
        # BRINGUP: request-driven fps would thread through the request API; the registered t2v preset
        # uses fps=24 (the fastvideo stage likewise defaults to a fixed fps for t2v). Held on the adapter
        # so the loop's dit-call stays ``dit(latent, text_embed, sigma)`` (toy-compatible — no fps kwarg).
        self.fps = int(fps)

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        hs = self._t(latent)  # [1, 16, T, h, w] — the RAW latent (no concat)
        ehs = self._t(text_embed)  # [1, seq, 100352]
        _b, _c, t, h, w = hs.shape
        s = float(sigma)
        # (1) per-frame 2D timestep [B, T] of the PLAIN sigma (NOT sigma*1000). For t2v every frame shares
        # the same sigma; the conditioned (V2W) path would override selected frames (BRINGUP, see loop).
        timestep = torch.full((1, t), s, device=self.device, dtype=torch.float32)
        # (2) mandatory masks: zeros condition_mask (no frame conditioned in t2v) + ones padding_mask
        # (no padding in t2v). The model resizes padding_mask to [h, w] and concats both internally.
        condition_mask = torch.zeros(1, 1, t, h, w, device=self.device, dtype=self.dtype)
        padding_mask = torch.ones(1, 1, h, w, device=self.device, dtype=self.dtype)
        # (3) fps scalar for RoPE temporal NTK scaling.
        fps_tensor = torch.tensor([float(self.fps)], device=self.device, dtype=self.dtype)
        with self._ctx():
            velocity = self.module(hidden_states=hs,
                                   timestep=timestep,
                                   encoder_hidden_states=ehs,
                                   fps=fps_tensor,
                                   condition_mask=condition_mask,
                                   padding_mask=padding_mask,
                                   return_dict=False)[0]
        return self._n(velocity)  # rectified-flow velocity (loop integrates it)


class Cosmos25WanVAE(TorchComponent):
    """Cosmos2.5 Wan-style causal 3D VAE (``fastvideo.models.vaes.cosmos25wanvae:Cosmos25WanVAE``).

    Its ``encode``/``decode`` ALREADY apply the (z-mean)/std normalization internally (the Cosmos latent
    contract), so this adapter only marshals numpy<->torch — it must NOT re-normalize (that is why we do
    not reuse the v2 ``WanVAE`` adapter, which would double-apply the per-channel stats)."""

    @torch.no_grad()
    def decode(self, latent):
        z = self._t(latent)  # [1, 16, T, h, w] — a NORMALIZED latent
        video = self.module.decode(z.to(self.dtype))  # -> video [B, 3, T, H, W] in [-1, 1]
        if hasattr(video, "sample"):
            video = video.sample
        return self._n(video)

    @torch.no_grad()
    def encode(self, video):
        dist = self.module.encode(self._t(video))
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        return self._n(z)  # already the NORMALIZED latent the DiT expects


class Cosmos25Reason1Encoder(TorchComponent):
    """Reason1 (Qwen2.5-VL) text encoder. ``encode(text) -> embedding[seq, 100352]`` (numpy out).

    Faithful to ``Cosmos25TextEncodingStage``: call ``compute_text_embeddings_online({"text": [text]},
    "text")`` (the encoder applies the chat template + per-layer mean-normalize + full-concat itself, so
    we do NOT re-implement the 100352-dim normalization here). The negative prompt is encoded the same
    way for ClassicCFG. The encoder is multimodal/heavy — BRINGUP, GPU-only.

    The ``tokenizer`` extra from ``_make_text_encoder`` is accepted for signature compatibility but
    unused: ``Reason1TextEncoder`` owns its own processor/tokenizer (``self.processor.tokenizer``)."""

    def __init__(self, module, tokenizer=None, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer  # unused; Reason1 carries its own processor (BRINGUP note above)

    @torch.no_grad()
    def encode(self, text):
        with self._ctx():
            # returns [B, seq, 100352]; single-request B=1 -> squeeze to [seq, 100352] (the DiT adapter's
            # _t re-adds the batch dim, matching the fastvideo stage's batch.prompt_embeds[0]).
            embeds = self.module.compute_text_embeddings_online({"text": [text or ""]}, "text")
        if hasattr(embeds, "shape") and len(embeds.shape) == 3:
            embeds = embeds[0]
        return _to_numpy(embeds)
