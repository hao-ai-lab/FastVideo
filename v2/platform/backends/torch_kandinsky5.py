"""Kandinsky-5.0 torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter`` so
the recipe is self-contained (no edit to the shared ``_make_dit``/``_make_vae``/``_make_text_encoder``
dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only on a GPU box.

Faithful to the diffusers ``KandinskyV5...Pipeline`` (vendored under
``diffusers/pipelines/kandinsky5/pipeline_kandinsky.py``) and ``fastvideo.models.dits.kandinsky5``:

  * ``Kandinsky5DiT`` — velocity predictor. The loop hands a CHANNELS-FIRST ``latent[C,T,H,W]`` (Wan /
    ToyDiT convention), the Qwen token embeds as ``text_embed``, the CLIP pooled vector via ``context``,
    and ``sigma``; this adapter PERMUTES to the channels-LAST ``[B,T,H,W,C]`` the forward requires (and
    permutes the velocity back). It also builds the THREE per-request positional inputs INTERNALLY from
    the latent geometry + recovered resolution: ``visual_rope_pos`` over the patched grid
    (patch_size=(1,2,2) -> ``[arange(T), arange(H//2), arange(W//2)]``), ``text_rope_pos`` over the Qwen
    seq len, and the resolution-dependent ``scale_factor`` (= (1,2,2) for 480–854 else (1,3.16,3.16)).
    ``timestep = σ·1000``. ``pooled_projections`` is MANDATORY (the forward raises if None);
    ``sparse_params=None`` (the Lite checkpoint uses regular attention — the sparse/STA path raises
    NotImplementedError in fastvideo, BRINGUP risk K).
  * ``Kandinsky5QwenEncoder`` — Qwen2.5-VL. Applies the repo prompt template + start-idx slicing
    (``prompt_template_encode_start_idx=129``, ``max_sequence_length=256``): the embeds are the LAST
    hidden state (``output_hidden_states=True``) sliced from index 129 onward (BRINGUP risk G).
  * ``Kandinsky5ClipEncoder`` — CLIP-with-projection. Returns the ``pooler_output`` (already projected to
    768) for a 77-token padded prompt. Its tokenizer lives in ``tokenizer_2`` (NOT the ``tokenizer`` the
    built-in maker probes), so the adapter re-resolves it from the model root (BRINGUP).
  * ``Kandinsky5VAE`` — AutoencoderKLHunyuanVideo. SCALAR ``scaling_factor`` (latent = raw·sf; decode raw
    = latent/sf), NOT Wan per-channel (z-mean)/std. The loop's latent is already channels-first
    ``[C,T,H,W]`` (the DiT adapter owns the channels-last conversion), so decode only adds a batch dim +
    un-normalizes by the scalar before ``module.decode`` (BRINGUP risk F).
"""
from __future__ import annotations

import os
from typing import Any

import torch

from v2.platform.backends.torch_backend import TorchComponent, _to_numpy

# Repo defaults (diffusers pipeline + fastvideo Kandinsky5ArchConfig). The two misspellings inside the
# template below are VERBATIM from the upstream Kandinsky-5 prompt template (diffusers
# pipeline_kandinsky.py) — they must match the tokens the model was trained on, so the codespell:ignore
# directives on those lines are deliberate; do not "fix" them.
_QWEN_PROMPT_TEMPLATE = "\n".join([
    "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",  # codespell:ignore promt
    "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
    "Describe the location of the video, main characters or objects and their action.",
    "Describe the dynamism of the video and presented actions.",
    "Name the visual style of the video: whether it is a professional footage, user generated content, "
    "some kind of animation, video game or scren content.",  # codespell:ignore scren
    "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
    "Pay attention to the order of key actions shown in the scene.<|im_end|>",
    "<|im_start|>user\n{}<|im_end|>",
])
_QWEN_ENCODE_START_IDX = 129
_QWEN_MAX_SEQ_LEN = 256
_CLIP_MAX_LEN = 77
_PATCH_SIZE = (1, 2, 2)  # transformer patch_size (T, H, W)
_SPATIAL_RATIO = 8  # VAE spatial compression (latent_h = H // 8)
_TIMESTEP_SCALE = 1000.0  # σ·1000 timestep convention


def _get_scale_factor(height: int, width: int) -> tuple[float, float, float]:
    """Resolution-dependent RoPE scaling (pipeline ``_get_scale_factor``): (1,2,2) when BOTH height and
    width are in [480, 854], else (1, 3.16, 3.16). BRINGUP risk E (must not be left at (1,1,1))."""

    def between_480p(x: int) -> bool:
        return 480 <= x <= 854

    if between_480p(height) and between_480p(width):
        return (1.0, 2.0, 2.0)
    return (1.0, 3.16, 3.16)


class Kandinsky5DiT(TorchComponent):
    """``dit(latent[C,T,H,W], qwen_embed[seq,3584], sigma, context=clip_pooled[768]) -> velocity[C,T,H,W]``.

    Real forward (kandinsky5.py): ``forward(hidden_states[B,T,H,W,C], encoder_hidden_states, timestep,
    pooled_projections, visual_rope_pos, text_rope_pos, scale_factor, sparse_params, return_dict)``. The
    loop's channels-FIRST latent is permuted to channels-LAST here and the velocity is permuted back."""

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        hs = self._t(latent)  # [1, C, T, H, W] (channels-first from the loop)
        hs = hs.permute(0, 2, 3, 4, 1).contiguous()  # -> [1, T, H, W, C] (the forward's geometry)
        ehs = self._t(text_embed)  # [1, seq, 3584]
        if context is None:
            raise ValueError("Kandinsky5DiT requires the CLIP pooled vector (passed as `context`); "
                             "pooled_projections is mandatory for the Kandinsky5 forward.")
        pooled = self._t(context)  # [1, 768]
        _b, t, h, w, _c = hs.shape
        ts = torch.tensor([float(sigma) * _TIMESTEP_SCALE], device=self.device, dtype=self.dtype)
        # visual_rope_pos over the PATCHED grid: patch_size (1,2,2) -> (T, H//2, W//2). Matches the
        # pipeline's [arange(num_latent_frames), arange(latent_h//2), arange(latent_w//2)].
        gh = max(1, h // _PATCH_SIZE[1])
        gw = max(1, w // _PATCH_SIZE[2])
        gt = max(1, t // _PATCH_SIZE[0])
        visual_rope_pos = [
            torch.arange(gt, device=self.device),
            torch.arange(gh, device=self.device),
            torch.arange(gw, device=self.device)
        ]
        text_rope_pos = torch.arange(ehs.shape[1], device=self.device)
        # Recover pixel resolution from the latent geometry for the resolution-dependent scale_factor.
        scale_factor = _get_scale_factor(h * _SPATIAL_RATIO, w * _SPATIAL_RATIO)
        with self._ctx():
            velocity = self.module(
                hidden_states=hs,
                encoder_hidden_states=ehs,
                timestep=ts,
                pooled_projections=pooled,
                visual_rope_pos=visual_rope_pos,
                text_rope_pos=text_rope_pos,
                scale_factor=scale_factor,
                sparse_params=None,  # Lite checkpoint = regular attention (BRINGUP K)
                return_dict=False)
        if isinstance(velocity, tuple):
            velocity = velocity[0]
        velocity = velocity.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,H,W,C] -> channels-first [B,C,T,H,W]
        return self._n(velocity)  # channels-FIRST rectified-flow velocity


class Kandinsky5QwenEncoder(TorchComponent):
    """Qwen2.5-VL text encoder. ``encode(text) -> token embeds[seq, 3584]`` after the prompt template +
    start-idx (129) slice on the LAST hidden state. The built-in text-encoder maker passes the
    ``tokenizer`` subfolder tokenizer (the Qwen one) — correct here."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = _QWEN_MAX_SEQ_LEN):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, text):
        full = _QWEN_PROMPT_TEMPLATE.format(text or "")
        max_allowed = _QWEN_ENCODE_START_IDX + self.max_length
        toks = self.tokenizer(text=[full],
                              images=None,
                              videos=None,
                              max_length=max_allowed,
                              truncation=True,
                              return_tensors="pt",
                              padding=True)
        ids = toks["input_ids"].to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, return_dict=True, output_hidden_states=True)
        # The pipeline uses hidden_states[-1] (the final hidden state) sliced from the start idx onward.
        hidden = out["hidden_states"][-1] if isinstance(out, dict) else out.hidden_states[-1]
        hidden = hidden[:, _QWEN_ENCODE_START_IDX:].squeeze(0)  # [seq, 3584]
        return _to_numpy(hidden)


class Kandinsky5ClipEncoder(TorchComponent):
    """CLIP-with-projection text encoder. ``encode(text) -> pooled[768]``. The built-in maker hands this
    adapter the wrong (``tokenizer/`` = Qwen) tokenizer, so we re-resolve the CLIP tokenizer from the
    ``tokenizer_2`` sibling of the model root (BRINGUP: confirm the subfolder name on a real checkpoint)."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = _CLIP_MAX_LEN):
        super().__init__(module, device=device, dtype=dtype)
        self.max_length = max_length
        self.tokenizer = self._resolve_clip_tokenizer(module, tokenizer)

    @staticmethod
    def _resolve_clip_tokenizer(module: Any, fallback: Any) -> Any:
        # The model root holds text_encoder_2/ (this module) and tokenizer_2/ (its tokenizer). Recover it
        # from the module config's _name_or_path when available; else fall back to the passed tokenizer.
        from transformers import AutoTokenizer
        name = getattr(getattr(module, "config", None), "_name_or_path", "") or ""
        if name:
            root = os.path.dirname(os.path.normpath(name))  # .../text_encoder_2 -> root
            tok2 = os.path.join(root, "tokenizer_2")
            if os.path.isdir(tok2):
                return AutoTokenizer.from_pretrained(tok2)
        return fallback

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "",
                              max_length=self.max_length,
                              truncation=True,
                              add_special_tokens=True,
                              padding="max_length",
                              return_tensors="pt")
        ids = toks["input_ids"].to(self.device)
        mask = toks.get("attention_mask")
        kw = {"attention_mask": mask.to(self.device)} if mask is not None else {}
        with self._ctx():
            out = self.module(input_ids=ids, **kw)
        pooled = out["pooler_output"] if isinstance(out, dict) else out.pooler_output
        return _to_numpy(pooled.squeeze(0))  # [768]


class Kandinsky5VAE(TorchComponent):
    """AutoencoderKLHunyuanVideo with a SCALAR scaling_factor. The DiT works in scaled latent space
    (latent = raw·sf); decode inverts it (raw = latent/sf) — NOT the Wan per-channel (z-mean)/std. The
    loop latent is already channels-first ``[C,T,H,W]`` (the DiT adapter owns the channels-last view)."""

    def _scaling_factor(self) -> float:
        cfg = getattr(self.module, "config", None)
        sf = getattr(cfg, "scaling_factor", None) if cfg is not None else None
        return float(sf) if sf is not None else 0.476986

    @torch.no_grad()
    def decode(self, latent):
        z = self._t(latent).float()  # [1, C, T, H, W]
        z = z / self._scaling_factor()  # un-normalize (scalar)
        video = self.module.decode(z.to(self.dtype))
        if hasattr(video, "sample"):
            video = video.sample
        return self._n(video)  # [3, T, H*8, W*8] in [-1, 1]

    @torch.no_grad()
    def encode(self, video):
        x = self._t(video)
        dist = self.module.encode(x)
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        return self._n(z.float() * self._scaling_factor())  # raw -> scaled latent the DiT consumes (channels-first)
