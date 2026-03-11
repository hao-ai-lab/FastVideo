# SPDX-License-Identifier: Apache-2.0
"""Text embedding utilities for RL training."""

from __future__ import annotations

import torch


def compute_text_embeddings(
    prompts: list[str],
    text_encoder,
    tokenizer,
    max_sequence_length: int = 512,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Encode text prompts into embeddings using T5.

    Args:
        prompts: List of text prompts.
        text_encoder: T5 text encoder model.
        tokenizer: T5 tokenizer.
        max_sequence_length: Max token length.
        device: Target device.

    Returns:
        Tensor of shape (B, L, D).
    """
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids, attention_mask=attention_mask
        ).last_hidden_state
    
    # make padding token 0
    prompt_embeds[attention_mask == 0] = 0
    return prompt_embeds
