# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 prompt tokenization parity — chat template + special tokens (Tier A).

Reference: ``vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py``
lines 562-606 (``Cosmos3OmniDiffusersPipeline._tokenize_prompt``).

The Cosmos3 tokenizer is a Qwen2 chat-template tokenizer with two
appended special tokens:

  * ``eos_token_id`` == 151645
  * ``<|vision_start|>`` == 151652

The reference pipeline:
  1. Wraps the prompt in a ``role=user`` conversation (optionally with a
     ``role=system`` prefix carrying ``COSMOS3_SYSTEM_PROMPT``);
  2. Applies the chat template with ``add_generation_prompt=True``;
  3. Truncates to ``max_sequence_length``;
  4. Appends ``eos`` and ``<|vision_start|>``;
  5. Right-pads with ``pad_token_id`` and produces a matching
     attention mask of ``[1]*seq_len + [0]*pad_len``.

This Tier A scaffold validates the two special-token IDs and the
right-pad shape contract. Full byte-for-byte parity against an
official Cosmos3 tokenizer requires Phase 2b weights.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.local]


def test_cosmos3_special_token_ids() -> None:
    """Asserts the two Cosmos3 special-token IDs the pipeline depends on.

    Specifically ``eos_token_id == 151645`` and
    ``convert_tokens_to_ids('<|vision_start|>') == 151652``. These IDs
    are pinned by the Qwen2 base tokenizer and are appended in
    ``Cosmos3OmniDiffusersPipeline._tokenize_prompt`` at
    pipeline_cosmos3.py:596-597.
    """
    try:
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (  # type: ignore
            Cosmos3OmniDiffusersPipeline,
        )
    except ImportError:
        pytest.skip("FastVideo Cosmos3 pipeline not yet implemented (Phase 2b)")

    pipeline = Cosmos3OmniDiffusersPipeline.__new__(Cosmos3OmniDiffusersPipeline)
    tokenizer = getattr(pipeline, "tokenizer", None)
    if tokenizer is None:
        pytest.skip("Cosmos3 tokenizer instance not yet wired on the pipeline")

    assert tokenizer.eos_token_id == 151645
    assert tokenizer.convert_tokens_to_ids("<|vision_start|>") == 151652


def test_cosmos3_chat_template_shape_contract() -> None:
    """Asserts the right-pad shape contract of ``_tokenize_prompt``.

    For ``max_sequence_length = N``, the returned ``input_ids`` and
    ``attention_mask`` must each be ``[1, N]``, with the first
    ``seq_len`` mask entries equal to 1 and the remainder equal to 0.

    Once Phase 2b lands, this test should:
      * call ``_tokenize_prompt`` directly with a small max_seq_length;
      * assert ``input_ids.shape == (1, max_seq_length)``;
      * assert ``attention_mask.sum().item() == seq_len`` and that the
        last two non-pad tokens are ``[eos, vision_start]``.
    """
    try:
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (  # type: ignore
            Cosmos3OmniDiffusersPipeline,
        )
    except ImportError:
        pytest.skip("FastVideo Cosmos3 pipeline not yet implemented (Phase 2b)")

    pipeline = Cosmos3OmniDiffusersPipeline.__new__(Cosmos3OmniDiffusersPipeline)
    if not hasattr(pipeline, "_tokenize_prompt"):
        pytest.skip("_tokenize_prompt not yet wired on the FastVideo pipeline")

    input_ids, attention_mask = pipeline._tokenize_prompt(
        "A robot.", max_sequence_length=32, use_system_prompt=False
    )
    assert tuple(input_ids.shape) == (1, 32)
    assert tuple(attention_mask.shape) == (1, 32)
    seq_len = int(attention_mask.sum().item())
    assert attention_mask[0, :seq_len].sum().item() == seq_len
    assert attention_mask[0, seq_len:].sum().item() == 0
    assert int(input_ids[0, seq_len - 2].item()) == 151645
    assert int(input_ids[0, seq_len - 1].item()) == 151652
