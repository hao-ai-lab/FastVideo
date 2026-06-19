"""Cosmos-Predict2 torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter``
so the Cosmos recipe is self-contained (no edit to the shared ``_make_dit``/``_make_text_encoder``
dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only on a GPU box.

* ``CosmosDiT`` — the EDM denoiser ``F_θ``. The loop hands the *already EDM-input-scaled* model input
  (``x·c_in``) and the model timestep ``t = σ·1000``; this adapter returns the **raw** transformer
  output (the EDM ``c_skip``/``c_out`` → x0 reconstruction + x0-space CFG live in ``CosmosDenoiseLoop``,
  NOT here). It builds the mandatory zero ``condition_mask`` / ``padding_mask`` and ``fps`` the Cosmos
  forward requires (the model concats the masks internally → 18ch patch_embed input). Faithful to
  ``CosmosDenoisingStage._run_transformer`` (fastvideo/pipelines/stages/denoising.py).
* ``CosmosT5Encoder`` — T5-Large (1024-dim). Cosmos uses the **raw** last_hidden_state (NaN→0, no
  fixed-length zero-pad), unlike the Wan T5 convention that zero-pads to 512.
"""
from __future__ import annotations

import torch

from v2.platform.backends.torch_backend import T5Encoder, TorchComponent, _to_numpy


class CosmosDiT(TorchComponent):
    """``dit(model_input[C,T,h,w], text_embed[seq,1024], timestep) -> raw noise_pred[C,T,h,w]``.

    ``model_input`` is pre-scaled by the loop (``x·c_in``); ``timestep`` is ``σ·1000`` (1D [B] float)."""

    @torch.no_grad()
    def __call__(self, model_input, text_embed, timestep, context=None, *, cond=None):
        hs = self._t(model_input)  # [1, C, T, h, w]
        ehs = self._t(text_embed)
        b, _c, t, h, w = hs.shape
        ts = torch.tensor([float(timestep)], device=self.device, dtype=self.dtype).expand(b)
        condition_mask = torch.zeros(b, 1, t, h, w, device=self.device, dtype=self.dtype)  # t2v: zeros
        padding_mask = torch.zeros(1, 1, h, w, device=self.device, dtype=self.dtype)  # resized→[h,w]
        with self._ctx():
            out = self.module(hidden_states=hs,
                              timestep=ts,
                              encoder_hidden_states=ehs,
                              fps=24,
                              condition_mask=condition_mask,
                              padding_mask=padding_mask,
                              return_dict=False)[0]
        return self._n(out)  # RAW EDM output (loop reconstructs x0)


class CosmosT5Encoder(T5Encoder):
    """Cosmos T5-Large: raw last_hidden_state (NaN→0), NO zero-pad-to-max (the Wan convention would
    mis-condition Cosmos). Truncates to ``max_length`` but returns only the real-token rows."""

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "", return_tensors="pt", max_length=self.max_length, truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask)
        hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).squeeze(0)
        hidden = torch.nan_to_num(hidden, nan=0.0)
        n = int(mask.sum().item())  # zero any row beyond real-token length
        if n < hidden.shape[0]:
            hidden[n:] = 0.0
        return _to_numpy(hidden)
