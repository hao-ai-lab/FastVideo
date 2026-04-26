# SPDX-License-Identifier: Apache-2.0
"""Per-rank negative-prompt encoding shared by training model plugins.

Encoding the negative prompt only on rank 0 and broadcasting (the
previous Wan path) ran ``Pipeline.from_pretrained`` asymmetrically across
ranks, which deadlocked on any collective fired during text-encoder load
(FSDP device-mesh init, weight broadcast, etc.). The text encoder is
small and only loaded once at startup, so loading it on every rank
sidesteps the deadlock entirely.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer

from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TextEncoderLoader
from fastvideo.train.utils.moduleloader import make_inference_args
from fastvideo.utils import maybe_download_model

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig


def encode_negative_prompt(
    training_config: TrainingConfig,
    *,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    encoder_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-rank encode of ``prompt`` using encoder ``encoder_index``.

    Reads ``pipeline_config.text_encoder_configs[encoder_index]`` so the
    encoder class (e.g. UMT5 for Wan) and tokenizer kwargs match the
    inference path, and applies the matching ``postprocess_text_funcs``
    entry. Returns ``(embeds, mask)`` on ``device`` cast to ``dtype``.
    """
    tc = training_config
    pipeline_config = tc.pipeline_config
    if pipeline_config is None:
        raise ValueError("training_config.pipeline_config is required for negative "
                         "prompt encoding")

    encoder_configs = pipeline_config.text_encoder_configs
    postprocess_funcs = pipeline_config.postprocess_text_funcs
    preprocess_funcs = getattr(pipeline_config, "preprocess_text_funcs", None)

    if encoder_index < 0 or encoder_index >= len(encoder_configs):
        raise IndexError(f"encoder_index {encoder_index} out of range for "
                         f"text_encoder_configs (len={len(encoder_configs)})")
    encoder_config = encoder_configs[encoder_index]
    postprocess_text = postprocess_funcs[encoder_index]
    preprocess_text = (preprocess_funcs[encoder_index] if preprocess_funcs is not None else None)

    # HF convention: text_encoder / tokenizer for index 0,
    # text_encoder_2 / tokenizer_2 for index 1, etc.
    suffix = "" if encoder_index == 0 else f"_{encoder_index + 1}"
    encoder_subdir = f"text_encoder{suffix}"
    tokenizer_subdir = f"tokenizer{suffix}"

    model_path = maybe_download_model(tc.model_path)
    inference_args = make_inference_args(tc, model_path=model_path)
    # Keep the encoder on-device; CPU offload would init an FSDP device
    # mesh and reintroduce the collective at load time.
    inference_args.text_encoder_cpu_offload = False

    loader = TextEncoderLoader()
    text_encoder = loader.load(
        os.path.join(model_path, encoder_subdir),
        inference_args,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, tokenizer_subdir))

    tok_kwargs = dict(encoder_config.tokenizer_kwargs)
    text = preprocess_text(prompt) if preprocess_text is not None else prompt

    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
    ):
        text_inputs = tokenizer(text, **tok_kwargs).to(device)
        outputs = text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )
        # Mirror TextEncodingStage: postprocess reads outputs.attention_mask.
        outputs.attention_mask = text_inputs["attention_mask"]
        embeds = postprocess_text(outputs).to(device=device, dtype=dtype)
        mask = text_inputs["attention_mask"].to(device=device, dtype=dtype)

    del text_encoder, tokenizer

    return embeds, mask
