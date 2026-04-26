# SPDX-License-Identifier: Apache-2.0
"""Stable Audio variant of `TextEncodingStage`.

Stable Audio's reference (`diffusers.StableAudioPipeline.encode_prompt`)
tokenizes with `padding="max_length"` and `max_length=tokenizer.model_max_length`
so positive + negative embeds end up the same shape, and the attention
mask is forwarded into the projection step. Mirror those two properties:

  1. Force `padding="max_length"` so positive and negative T5 outputs
     come out the same sequence length.
  2. Pre-initialize `prompt_attention_mask` / `negative_attention_mask`
     to `[]` so the parent stage actually populates them (the base
     `TextEncodingStage` only appends if the list is already non-`None`).
"""
from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage


class StableAudioTextEncodingStage(TextEncodingStage):

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.prompt_attention_mask is None:
            batch.prompt_attention_mask = []
        if batch.negative_attention_mask is None:
            batch.negative_attention_mask = []
        return super().forward(batch, fastvideo_args)

    def encode_text(self, text, fastvideo_args, **kwargs):
        kwargs.setdefault("padding", "max_length")
        # Stable Audio's HF repo ships the T5 tokenizer with
        # `model_max_length=128`, **not** the standard 512. The base
        # `T5Config.text_len` defaults to 512, which silently over-pads
        # by 4× and pollutes the DiT's cross-attention with extra
        # bias-only KV positions, drifting CFG outputs hard. Force the
        # tokenizer's own max length so we mirror diffusers exactly.
        if "max_length" not in kwargs and self.tokenizers:
            mml = getattr(self.tokenizers[0], "model_max_length", None)
            if mml is not None and mml > 0:
                kwargs["max_length"] = mml
        return super().encode_text(text, fastvideo_args, **kwargs)
