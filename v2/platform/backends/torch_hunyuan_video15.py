"""HunyuanVideo 1.5 torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter``
so the recipe is self-contained (no edit to the shared ``_make_dit``/``_make_vae``/``_make_text_encoder``
dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only on a GPU box.

Architecture deltas vs Wan that force NEW adapters (all faithful to
``fastvideo/models/dits/hunyuanvideo15.py`` + ``fastvideo/pipelines/stages/denoising.py``):

* ``HunyuanVideo15DiT`` — the rectified-flow velocity network ``F_θ``. Unlike ``WanDiT`` (single text
  embed, single image embed), the real ``forward`` takes ``encoder_hidden_states`` as a **LIST of two**
  tensors ``[qwen_hidden(3584-d), byt5_hidden(1472-d)]`` and ``encoder_hidden_states_image`` as a
  **LIST of one** image-embed tensor (all-zeros ``[1,729,1152]`` for t2v → the DiT's
  ``torch.all(image_embeds==0)`` auto-selects the t2v token-ordering branch). The loop hands the two
  text embeds as a tuple in ``text_embed`` and the image embeds in ``context``; this adapter marshals
  them into the two lists internally and builds the zero image-embed default, so the loop's dit-call
  signature stays ``dit(latent, text_embed, sigma, context=, cond=)`` (toy-compatible). For i2v the
  loop passes the 33-channel first-frame conditioning latent (32 VAE + 1 mask) in ``cond``; the adapter
  concats it onto the noisy latent on the channel dim to reach ``in_channels=65`` BEFORE the forward
  (mirrors ``denoising.py`` line 393's ``torch.cat([latent, image_latent], dim=1)``). ``timestep`` is
  ``sigma*1000`` (the scheduler's discrete timestep; ``scale_model_input`` is identity for flow-match).
  The network predicts velocity directly → returned via ``self._n()`` with NO x0→velocity conversion
  (unlike ``LTX2DiT``). ``use_meanflow`` SR variants pass ``timestep_r`` — out of scope here (t2v/i2v
  ship with ``use_meanflow=False``), so ``timestep_r=None``.
* ``HunyuanVideo15VAE`` — the HunyuanVideo 1.5 causal-3D VAE (z=32, 16× spatial / 4× temporal). Its DiT
  latent space is normalized by a single **scalar** ``scaling_factor=1.03682`` (NOT Wan's per-channel
  ``latents_mean``/``latents_std`` — the ``WanVAE`` adapter would corrupt these latents, BRINGUP risk D).
  Decode: ``module.decode(latent / scaling_factor)``; encode (i2v first-frame cond): ``z * scaling_factor``.
* ``HunyuanVideo15QwenEncoder`` — Qwen2.5-VL primary text encoder. Faithful to
  ``configs/pipelines/hunyuan15.py:qwen_preprocess_text/qwen_postprocess_text``: wraps the prompt in the
  system+chat template, requests ``output_hidden_states=True``, takes ``hidden_states[-3]`` (3rd-from-last
  layer, NOT ``last_hidden_state``), and crops the first 108 prompt-template tokens. The generic
  ``T5Encoder.encode`` (last_hidden_state, pad-to-max) is WRONG for Qwen (BRINGUP risk B).
* ``HunyuanVideo15ByT5Encoder`` — ByT5/Glyph secondary encoder. ``byt5_preprocess_text`` extracts quoted
  glyph texts via regex; when the prompt has no quoted text it is ``None`` and the real pipeline emits a
  zero-length ``[1, 0, 1472]`` embedding (the DiT always concatenates the txt_in_2 output) — this adapter
  reproduces that (BRINGUP risk C). Output is the raw ``last_hidden_state``.

All forwards run inside ``self._ctx()`` (FastVideo attention reads attn_metadata via the forward
context). BRINGUP: the DiT's ``sequence_model_parallel_shard``/all-gather is a no-op at sp=1 (the card
defaults parallelism to single() until SP is validated — risk G).
"""
from __future__ import annotations

import re

import numpy as np
import torch

from v2.platform.backends.torch_backend import (
    NUM_TRAIN_TIMESTEPS,
    T5Encoder,
    TorchComponent,
    _to_numpy,
)

# Faithful constants from configs/pipelines/hunyuan15.py + Hy15ImageEncodingStage.
HUNYUAN15_VAE_SCALING_FACTOR = 1.03682  # scalar (NOT per-channel mean/std)
HUNYUAN15_IMAGE_TOKENS = 729  # SigLIP token count placeholder
HUNYUAN15_IMAGE_DIM = 1152  # config.image_embed_dim
HUNYUAN15_BYT5_DIM = 1472  # config.text_embed_2_dim
PROMPT_TEMPLATE_TOKEN_LENGTH = 108  # tokens of the chat-template prefix to crop off Qwen
QWEN_HIDDEN_STATE_INDEX = -3  # qwen_postprocess takes hidden_states[-3]

PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    "        1. The main content and theme of the video. "
    "        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    "        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    "        4. background environment, light, style and atmosphere. "
    "        5. camera angles, movements, and transitions used in the video.")

# byt5_preprocess_text: pull the quoted glyph strings out of the prompt (HunyuanVideo 1.5 glyph rendering).
_GLYPH_PATTERN = r"\"(.*?)\"|“(.*?)”"


def _extract_glyph_texts(prompt: str) -> str | None:
    """Faithful to ``configs/pipelines/hunyuan15.py:extract_glyph_texts`` — returns ``None`` when the
    prompt carries no quoted text (the common t2v case), else a ``Text "..."``-joined glyph string."""
    matches = re.findall(_GLYPH_PATTERN, prompt or "")
    result = [m[0] or m[1] for m in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result
    return ". ".join([f'Text "{t}"' for t in result]) + ". " if result else None


class HunyuanVideo15DiT(TorchComponent):
    """``dit(latent[C,T,H,W], text_embed=(qwen,byt5), sigma, context=img_embeds, cond=image_latent)``.

    Marshals the two text embeds + the (zero-for-t2v) image embeds into the LIST args the real
    ``HunyuanVideo15Transformer3DModel.forward`` expects, concats the 33ch i2v cond onto the latent, and
    returns the rectified-flow velocity (NO x0 conversion). Faithful to ``denoising.py`` (timestep =
    sigma*1000, identity ``scale_model_input``, ``latent_model_input = cat([latent, image_latent])``)."""

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        hs = self._t(latent)  # [1, C(32), T, H, W]
        if cond is not None:  # i2v: concat first-frame cond latent (33ch) -> 65ch
            hs = torch.cat([hs, self._t(cond)], dim=1)
        # ``text_embed`` is a (qwen_hidden, byt5_hidden) tuple from the loop. Both are already numpy
        # [seq, dim]; the byt5 element may be a zero-length [0, 1472] (no glyph text).
        qwen_e, byt5_e = text_embed if isinstance(text_embed, tuple | list) else (text_embed, None)
        qwen_t = self._t(qwen_e)  # [1, qseq, 3584]
        if byt5_e is None:
            byt5_t = torch.zeros(1, 0, HUNYUAN15_BYT5_DIM, device=self.device, dtype=self.dtype)
        else:
            byt5_t = self._t(byt5_e)  # [1, bseq, 1472]
        # encoder_hidden_states_image: zeros for t2v (the DiT's ``torch.all(==0)`` selects the t2v branch).
        if context is None:
            img_t = torch.zeros(1, HUNYUAN15_IMAGE_TOKENS, HUNYUAN15_IMAGE_DIM, device=self.device, dtype=self.dtype)
        else:
            img_t = self._t(context)
        ts = float(sigma) * NUM_TRAIN_TIMESTEPS  # the scheduler's discrete timestep (BRINGUP risk B)
        timestep = torch.tensor([ts], device=self.device, dtype=self.dtype).expand(hs.shape[0])
        with self._ctx():
            velocity = self.module(hidden_states=hs,
                                   encoder_hidden_states=[qwen_t, byt5_t],
                                   timestep=timestep,
                                   encoder_hidden_states_image=[img_t],
                                   guidance=None,
                                   timestep_r=None,
                                   attention_kwargs=None)
        return self._n(velocity)  # rectified-flow velocity (BRINGUP risk C)


class HunyuanVideo15VAE(TorchComponent):
    """Scalar-scaling VAE. Decode un-scales (``latent / scaling_factor``) before ``module.decode``; encode
    (i2v first-frame cond) scales the latent (``z * scaling_factor``). NO per-channel mean/std (risk D)."""

    @torch.no_grad()
    def decode(self, latent):
        z = self._t(latent).float() / HUNYUAN15_VAE_SCALING_FACTOR
        video = self.module.decode(z.to(self.dtype))  # -> video [B,3,T,H,W] in [-1,1]
        return self._n(video)

    @torch.no_grad()
    def encode(self, video):
        x = self._t(video)
        dist = self.module.encode(x)
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        return self._n(z.float() * HUNYUAN15_VAE_SCALING_FACTOR)


class HunyuanVideo15QwenEncoder(TorchComponent):
    """Qwen2.5-VL primary text encoder. ``encode(text) -> embedding[seq, 3584]`` (numpy).

    Faithful to qwen_preprocess/postprocess: chat-template wrap, ``output_hidden_states=True``, take
    ``hidden_states[-3]``, crop the first 108 template tokens. The HF tokenizer's ``apply_chat_template``
    builds the prompt; ``max_length`` defaults to the pipeline's 1000+108 (risk B)."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = 1000 + PROMPT_TEMPLATE_TOKEN_LENGTH):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, text):
        # Qwen2_5_VLProcessor wraps an inner ``.tokenizer``; fall back to the tokenizer itself.
        tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        messages = [{
            "role": "system",
            "content": PROMPT_TEMPLATE_ENCODE_VIDEO
        }, {
            "role": "user",
            "content": text if text else " "
        }]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(formatted, return_tensors="pt", max_length=self.max_length, truncation=True)
        ids = enc.input_ids.to(self.device)
        mask = enc.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hidden = out.hidden_states[QWEN_HIDDEN_STATE_INDEX]  # 3rd-from-last layer (NOT last)
        hidden = hidden[:, PROMPT_TEMPLATE_TOKEN_LENGTH:].squeeze(0)  # crop the chat-template prefix
        return _to_numpy(hidden)


class HunyuanVideo15ByT5Encoder(T5Encoder):
    """ByT5/Glyph secondary encoder. ``encode(text) -> embedding[seq, 1472]``. Extracts the quoted glyph
    text first (``byt5_preprocess_text``); when none, returns a zero-length ``[0, 1472]`` embedding — the
    real pipeline's special-case (the DiT always concats txt_in_2 output, risk C). Output is the raw
    ``last_hidden_state`` (NO Wan zero-pad-to-max)."""

    @torch.no_grad()
    def encode(self, text):
        glyph = _extract_glyph_texts(text)
        if glyph is None:  # no quoted text -> empty embedding
            return np.zeros((0, HUNYUAN15_BYT5_DIM), dtype="float32")
        toks = self.tokenizer(glyph, return_tensors="pt", max_length=self.max_length, truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask)
        hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).squeeze(0)
        return _to_numpy(torch.nan_to_num(hidden, nan=0.0))
