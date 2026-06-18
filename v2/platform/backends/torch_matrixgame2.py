"""Matrix-Game 2.0 torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter`` so
the recipe is self-contained (no edit to ``_make_dit``/``_make_image_encoder`` in ``torch_backend.py``).
Imported lazily by ``_explicit_adapter`` only on a GPU box.

Matrix-Game 2.0 is an INTERACTIVE world model: a distilled, causal/autoregressive, action-conditioned
Wan-family DiT. The deltas this adapter encodes vs the stock ``WanDiT`` (all faithful to
``fastvideo/models/dits/matrixgame2/causal_model.py:CausalMatrixGame2WanModel._forward_inference`` and
``fastvideo/pipelines/stages/matrixgame2_denoising.py:MatrixGame2CausalDenoisingStage``):

  * EPSILON output, NOT velocity — ``_forward_inference`` returns a bare velocity-shaped tensor the stage
    treats as predicted NOISE; the epsilon->x0 conversion (``pred_noise_to_pred_video`` =
    ``x - sigma_t*eps``) lives in ``MatrixGame2CausalDMDLoop``, NOT here. The adapter returns the raw model
    output.
  * CAUSAL KV-CACHE is mandatory — ``kwargs['kv_cache'] is not None`` selects ``_forward_inference`` (else
    ``_forward_train``). The DiT MUTATES kv_cache / crossattn_cache / kv_cache_mouse / kv_cache_keyboard in
    place (sliding-window eviction by ``local_attn_size``). The loop owns these cache lists across blocks
    and steps and hands them in via ``MatrixGame2CausalDiT.call(...)``; the loop keeps them in
    ``LoopState`` so interleaving stays safe.
  * PER-FRAME timestep ``[B, num_frames]`` (LongTensor 0..1000), NOT a scalar ``sigma*1000`` — the
    condition_embedder unflattens ``timestep_proj`` by (batch, post_patch_num_frames); a scalar breaks the
    reshape. The loop passes a numpy per-frame timestep array; this adapter keeps it 2-D and ``long``.
  * i2v via CHANNEL-CONCAT read off ``get_forward_context().forward_batch.image_latent`` — the model
    concats the conditioning latent onto ``hidden_states`` along dim=1 INTERNALLY (patch_embedding is the
    32-in checkpoint). So the adapter places ``image_latent`` (the raw cond_latent, NO 4ch Wan mask — this
    differs from Wan2.1 i2v) on a lightweight forward_batch under ``set_forward_context``.
  * CLIP image embeds (257x1280) are the SOLE cross-attention context (``encoder_hidden_states_image``);
    text is ignored (Matrix-Game 2.0 has no text encoder), but the forward still wants a placeholder
    ``encoder_hidden_states`` list.
  * ACTION conditioning (mouse[B,F,2] / keyboard[B,F,4]) + its own KV caches is the interactive surface.
    BRINGUP: the degenerate (no-action) path runs with mouse/keyboard=None; live action routing needs a
    request-API extension (see the recipe's loop docstring).

BRINGUP (written-not-run; GPU-only): importing ``causal_model`` triggers a flex_attention
``torch.compile`` and the inference path uses plain SDPA over the cache — both GPU-only. The exact cache
shapes (``local_attn_size`` window, mouse-cache batch dim ``B*frame_seq_len``) come from the loaded
arch_config and must be GPU-verified against a real distilled checkpoint.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from v2.platform.backends.torch_backend import TorchComponent, _to_numpy

# Matrix-Game 2.0 ships a Wan-VAE first-frame i2v conditioning latent that the DiT channel-concats; this is
# carried on a minimal forward_batch object the model reads via ``get_forward_context().forward_batch``.


class _CondBatch:
    """Minimal stand-in for fastvideo's ``ForwardBatch`` carrying only the field the causal DiT reads:
    ``image_latent`` (the i2v cond_concat). The real ForwardBatch has dozens of fields; the inference
    forward only touches ``getattr(batch, 'image_latent', None)``, so this is a faithful, tiny carrier."""

    def __init__(self, image_latent: Any) -> None:
        self.image_latent = image_latent


class MatrixGame2CausalDiT(TorchComponent):
    """``call(latent[C,T,h,w], image_embed[257,1280], timestep[T] (per-frame), kv_caches..., cond=...)``
    -> raw epsilon[C,T,h,w]``.

    The loop drives the causal block-autoregressive schedule and the DMD epsilon->x0 conversion; this
    adapter is a single ``_forward_inference`` call with the cache/i2v/action plumbing. It is NOT the toy
    ``dit(latent, text_embed, sigma)`` shape — the loop calls ``MatrixGame2CausalDiT.call`` explicitly with
    the causal kwargs (the toy backend exercises the SAME loop via its degenerate path; see the recipe
    loop). All arch-specifics are built INTERNALLY here so the loop stays backend-agnostic."""

    @torch.no_grad()
    def call(self,
             latent: np.ndarray,
             image_embed: Any,
             timestep: np.ndarray,
             *,
             kv_cache: list[dict] | None = None,
             crossattn_cache: list[dict] | None = None,
             kv_cache_mouse: list[dict] | None = None,
             kv_cache_keyboard: list[dict] | None = None,
             current_start: int = 0,
             start_frame: int = 0,
             num_frame_per_block: int = 1,
             mouse_cond: np.ndarray | None = None,
             keyboard_cond: np.ndarray | None = None,
             image_latent: np.ndarray | None = None,
             prompt_embeds: Any = None) -> np.ndarray:
        # hidden_states [B, C, T, h, w]; the DiT concats image_latent internally off the forward_batch.
        hs = self._t(latent)  # [1, C, T, h, w]
        b, _c, t, _h, _w = hs.shape
        # Per-frame integer timestep [B, num_frames] (long); keep 2-D — never collapse to a scalar.
        ts_np = np.asarray(timestep).reshape(-1)
        timestep_t = torch.as_tensor(ts_np, device=self.device, dtype=torch.long).reshape(1, -1).expand(b, t)
        # CLIP image embeds [257, 1280] are the SOLE cross-attn context; the model wants a list.
        img_ctx = self._t(image_embed) if image_embed is not None else None
        image_kwargs = {"encoder_hidden_states_image": [img_ctx] if img_ctx is not None else []}
        # Text is ignored but the forward wants a non-empty placeholder list (condition_embedder drops it).
        pe = self._t(prompt_embeds) if prompt_embeds is not None else None
        ehs: list[Any] = [pe] if pe is not None else []
        # Action conditioning (interactive surface) — None on the degenerate world-rollout path (BRINGUP).
        action_kwargs: dict[str, Any] = {"num_frame_per_block": int(num_frame_per_block)}
        if mouse_cond is not None:
            action_kwargs["mouse_cond"] = self._t(mouse_cond)
        if keyboard_cond is not None:
            action_kwargs["keyboard_cond"] = self._t(keyboard_cond)
        if kv_cache_mouse is not None:
            action_kwargs["kv_cache_mouse"] = kv_cache_mouse
        if kv_cache_keyboard is not None:
            action_kwargs["kv_cache_keyboard"] = kv_cache_keyboard
        # i2v cond_concat: the DiT reads image_latent off the forward_batch -> channel-concat dim=1.
        cond_lat = self._t(image_latent) if image_latent is not None else None
        from v2.forward_context import set_forward_context
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=_CondBatch(cond_lat)):
            eps = self.module(
                hs,
                ehs,
                timestep_t,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=int(current_start),
                start_frame=int(start_frame),
                **image_kwargs,
                **action_kwargs,
            )
        return self._n(eps)  # RAW epsilon (the loop converts eps -> x0 via the scheduler sigma table)


class MatrixGame2CLIPImageEncoder(TorchComponent):
    """CLIP vision encoder for Matrix-Game 2.0 i2v conditioning: ``encode_image(image) -> [257, 1280]`` (1
    CLS + 256 patch tokens), which the DiT's WanImageEmbedding projects to inner_dim and uses as the sole
    cross-attention context. Mirrors fastvideo's ``MatrixGame2ImageEncodingStage`` — the HF image processor
    (sibling ``image_processor`` subfolder) preprocesses, the encoder returns ``last_hidden_state``.
    BRINGUP: written-not-run; GPU-verify the processor/encoder subfolders + dtype against a real
    distilled checkpoint."""

    def __init__(self, module, processor, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)
        self.processor = processor

    @torch.no_grad()
    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._ctx():
            out = self.module(**inputs)
        embeds = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return _to_numpy(embeds.squeeze(0))
