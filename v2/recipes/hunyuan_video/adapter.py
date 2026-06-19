"""HunyuanVideo torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter`` so the
Hunyuan recipe is self-contained (no edit to the shared ``_make_dit``/``_make_vae``/``_make_text_encoder``
dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only on a GPU box.

Three deltas vs the Wan adapters:

* ``HunyuanVideoDiT`` — flow-match velocity DiT, BUT ``encoder_hidden_states`` is a **2-element list**
  ``[llama_hidden[B, L, 4096], clip_pooled[B, 768]]`` (the forward splits a list as
  ``txt = ehs[0]``, ``text_states_2 = ehs[1]``), NOT a single tensor like Wan. The loop hands the LLaMA
  per-token sequence as ``text_embed`` and the CLIP-pooled global vector via ``context=`` (Wan's existing
  conditioning channel — see ``recipes/hunyuan_video/loop.py``); this adapter assembles the pair
  internally. ``timestep = sigma*1000`` (scalar per-batch ``LongTensor``); the output is the **bare**
  velocity tensor (NO ``x0 -> velocity`` conversion, unlike LTX-2 — Hunyuan is velocity-native like Wan).
  ``guidance`` is left at the forward default: base HunyuanVideo has ``guidance_embeds=False`` so
  ``guidance_in is None`` and the arg is a no-op (BRINGUP: a guidance-distilled checkpoint with
  ``guidance_embeds=True`` would need ``embedded_cfg_scale`` wired in here).

* ``HunyuanVideoVAE`` — ``AutoencoderKLHunyuanVideo`` normalizes by a scalar ``scaling_factor`` (0.476986),
  NOT Wan's per-channel ``latents_mean``/``latents_std``. DiT latent = ``raw_z * scaling_factor`` on
  encode; decode divides ``latents / scaling_factor`` before ``module.decode``. Decode output is video in
  ``[-1, 1]``.

* ``HunyuanVideoLlamaEncoder`` / ``HunyuanVideoCLIPEncoder`` — the dual text path. LLaMA wraps the prompt
  in the ``PROMPT_TEMPLATE_ENCODE_VIDEO`` chat template, runs with ``output_hidden_states=True``, takes the
  intermediate hidden state ``hidden_states[-(2+1)]`` (skip the last 2 layers), and crops the first 95
  template tokens (``crop_start=95``). CLIP returns the ``pooler_output`` (768-dim global vector). Faithful
  to ``fastvideo/configs/pipelines/hunyuan.py`` (``llama_preprocess_text`` / ``llama_postprocess_text`` /
  ``clip_postprocess_text``).
"""
from __future__ import annotations

from typing import Any

import torch

from v2.platform.backends.torch_backend import (
    NUM_TRAIN_TIMESTEPS,
    TorchComponent,
    _to_numpy,
)

# The LLaMA prompt template + crop verbatim from fastvideo/configs/pipelines/hunyuan.py (recipe DATA, not
# model code, so v2 owns its own copy rather than importing the private pipeline symbol).
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>")
LLAMA_CROP_START = 95  # the template's fixed system-prompt token count (crop_start)
LLAMA_HIDDEN_STATE_SKIP_LAYER = 2  # use hidden_states[-(skip+1)], i.e. skip the last 2 LLaMA layers
LLAMA_TEXT_LEN = 256  # LlamaArchConfig.text_len (the tokenizer max_length before the template)
CLIP_TEXT_LEN = 77  # CLIPTextConfig.text_len


# --------------------------------------------------------------------------- #
# DiT adapter                                                                   #
# --------------------------------------------------------------------------- #
class HunyuanVideoDiT(TorchComponent):
    """``dit(latent[C,T,H,W], text_embed=llama_hidden[L,4096], sigma, context=clip_pooled[768])`` ->
    velocity[C,T,H,W]. Real forward (hunyuanvideo.py): ``forward(hidden_states[B,C,T,H,W],
    encoder_hidden_states=[llama, clip], timestep[B], guidance=None) -> velocity (bare tensor)``."""

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        hs = self._t(latent)  # [1, C, T, h, w]
        llama = self._t(text_embed)  # [1, L, 4096] per-token LLaMA sequence
        # timestep = sigma*1000 (the scalar per-batch convention the time_in TimestepEmbedder consumes).
        timestep = torch.tensor([float(sigma) * NUM_TRAIN_TIMESTEPS], device=self.device)
        if context is not None:  # the faithful 2-element list path
            clip_pooled = self._t(context)  # [1, 768] CLIP global vector
            encoder_hidden_states: object = [llama, clip_pooled]
        else:  # degenerate: a single tensor (token 0 pooled)
            encoder_hidden_states = llama
        with self._ctx():
            # guidance=None -> forward default 6016.0, but base guidance_in is None so it is ignored
            # (do NOT wire embedded_cfg_scale unless a guidance_embeds=True checkpoint is loaded). The DiT
            # does an internal sequence-parallel shard/all-gather that no-ops at sp=1 (the bring-up default).
            velocity = self.module(hidden_states=hs,
                                   encoder_hidden_states=encoder_hidden_states,
                                   timestep=timestep,
                                   guidance=None)
        return self._n(velocity)  # rectified-flow velocity (bare tensor)


# --------------------------------------------------------------------------- #
# VAE adapter                                                                   #
# --------------------------------------------------------------------------- #
class HunyuanVideoVAE(TorchComponent):
    """``AutoencoderKLHunyuanVideo`` uses a scalar ``scaling_factor`` (0.476986), not Wan's per-channel
    mean/std. DiT latent = ``raw_z * scaling_factor``; decode divides by it before ``module.decode``."""

    def _scaling_factor(self) -> float:
        # ParallelTiledVAE exposes ``scaling_factor`` as a property reading config.scaling_factor.
        sf = getattr(self.module, "scaling_factor", None)
        if sf is None:
            sf = getattr(getattr(self.module, "config", None), "scaling_factor", 0.476986)
        if sf is None:  # defensive: neither property nor config carried it
            sf = 0.476986
        return float(sf.item() if hasattr(sf, "item") else sf)

    @torch.no_grad()
    def encode(self, video):
        x = self._t(video)
        dist = self.module.encode(x)  # DiagonalGaussianDistribution
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        return self._n(z.float() * self._scaling_factor())  # -> the scaled latent the DiT denoises

    @torch.no_grad()
    def decode(self, latent):
        z = self._t(latent).float() / self._scaling_factor()  # invert the encode scaling -> raw latent
        video = self.module.decode(z.to(self.dtype))  # -> video [B,3,T,H,W] in [-1,1]
        return self._n(video)


# --------------------------------------------------------------------------- #
# Text encoders (dual: LLaMA per-token sequence + CLIP pooled global vector)    #
# --------------------------------------------------------------------------- #
class HunyuanVideoLlamaEncoder(TorchComponent):
    """LLaMA primary encoder: wrap the prompt in ``PROMPT_TEMPLATE_ENCODE_VIDEO``, run with
    ``output_hidden_states=True``, take ``hidden_states[-(skip+1)]`` (skip the last 2 layers) and crop the
    first 95 template tokens. Returns the per-token sequence [L, 4096] (the DiT ``txt`` / ``encoder_hidden_states[0]``).
    The module is loaded with ``output_hidden_states=True`` (HunyuanConfig.__post_init__ sets it)."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = LLAMA_TEXT_LEN):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, text):
        prompt = PROMPT_TEMPLATE_ENCODE_VIDEO.format(text or "")
        toks = self.tokenizer(prompt,
                              return_tensors="pt",
                              max_length=self.max_length + LLAMA_CROP_START,
                              truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("HunyuanVideoLlamaEncoder: module returned no hidden_states; load LLaMA with "
                               "output_hidden_states=True (HunyuanConfig.__post_init__ sets it).")
        hidden = hidden_states[-(LLAMA_HIDDEN_STATE_SKIP_LAYER + 1)]  # skip the last 2 layers
        hidden = hidden[:, LLAMA_CROP_START:]  # drop the template's 95 system tokens
        return _to_numpy(hidden.squeeze(0))


class HunyuanVideoCLIPEncoder(TorchComponent):
    """CLIP secondary encoder: returns ``pooler_output`` (the 768-dim global vector, the DiT
    ``text_states_2`` / ``encoder_hidden_states[1]``).

    The shared ``_make_text_encoder`` passes the ``tokenizer/`` subfolder (the LLaMA tokenizer) as the
    constructor's ``tokenizer`` arg; HunyuanVideo's CLIP tokenizer lives in ``tokenizer_2/``. So this
    adapter loads the CLIP tokenizer ITSELF from the sibling ``tokenizer_2/`` (derived from the LLaMA
    tokenizer's ``name_or_path`` = ``<root>/tokenizer``), ignoring the passed-in LLaMA tokenizer. On CPU
    this path is exercised by the ``ToyTextEncoder`` factory, which needs no tokenizer."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = CLIP_TEXT_LEN):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = self._load_clip_tokenizer(tokenizer)
        self.max_length = max_length

    @classmethod
    def _load_clip_tokenizer(cls, llama_tokenizer: Any) -> Any:
        """HunyuanVideo's CLIP tokenizer lives in ``tokenizer_2/`` (a CLIPTokenizer), but the shared
        ``_make_text_encoder`` only loads ``tokenizer/`` (the LLaMA tokenizer) and hands it to both
        encoders. Resolve the model root from the LLaMA tokenizer's ``name_or_path`` (= ``<root>/tokenizer``)
        and load the sibling ``tokenizer_2/`` with the right CLIP tokenizer class. Falls back to the
        passed-in tokenizer if the sibling can't be located (keeps the toy/CPU path working)."""
        import os
        name_or_path = getattr(llama_tokenizer, "name_or_path", "") or ""
        if not name_or_path:
            return llama_tokenizer
        tok2_dir = os.path.join(os.path.dirname(os.path.normpath(name_or_path)), "tokenizer_2")
        if not os.path.isdir(tok2_dir):
            return llama_tokenizer
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(tok2_dir)
        except TypeError:
            # transformers 5.x's slow CLIPTokenizer.__init__ calls
            # ``processors.RobertaProcessing(cls=..., sep=...)``, but tokenizers>=0.23 renamed those kwargs
            # (``cls`` -> positional ``cls_token``), so the slow load (and even the fast load, which
            # round-trips through it because tokenizer_2/ ships no tokenizer.json) raises TypeError. This is
            # an env library-version skew, not a checkpoint issue — build the CLIP fast tokenizer directly
            # from vocab.json+merges.txt (identical BPE/normalizer/pre-tokenizer/post-processor as CLIP, but
            # with the corrected positional RobertaProcessing call). Produces the standard CLIP ids
            # (bos=49406, eos=49407, pad=eos), matching ``CLIPTokenizer``.
            return cls._build_clip_fast_tokenizer(tok2_dir)

    @staticmethod
    def _build_clip_fast_tokenizer(tok2_dir: str) -> Any:
        import json
        import os

        from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
        from tokenizers.models import BPE
        from transformers import PreTrainedTokenizerFast

        with open(os.path.join(tok2_dir, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        merges = []
        with open(os.path.join(tok2_dir, "merges.txt"), encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if i == 0 and line.startswith("#"):
                    continue
                if line.strip():
                    a, _, b = line.partition(" ")
                    merges.append((a, b))
        bos, eos, unk, pad = "<|startoftext|>", "<|endoftext|>", "<|endoftext|>", "<|endoftext|>"
        tk = Tokenizer(
            BPE(vocab=vocab,
                merges=merges,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token=unk))
        tk.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "),
             normalizers.Lowercase()])
        tk.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(
                r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""),
                                 behavior="removed",
                                 invert=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ])
        tk.decoder = decoders.ByteLevel()
        bos_id, eos_id = tk.token_to_id(bos), tk.token_to_id(eos)
        tk.post_processor = processors.RobertaProcessing((eos, eos_id), (bos, bos_id),
                                                         add_prefix_space=False,
                                                         trim_offsets=False)
        return PreTrainedTokenizerFast(tokenizer_object=tk,
                                       bos_token=bos,
                                       eos_token=eos,
                                       unk_token=unk,
                                       pad_token=pad,
                                       model_max_length=CLIP_TEXT_LEN)

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "",
                              return_tensors="pt",
                              max_length=self.max_length,
                              truncation=True,
                              padding="max_length")
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask)
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out[1]
        return _to_numpy(pooled.squeeze(0))  # [768] global vector
