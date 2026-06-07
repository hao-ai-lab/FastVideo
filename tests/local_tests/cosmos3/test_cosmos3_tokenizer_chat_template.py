# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 prompt tokenization contract — chat template + special tokens.

The native pipeline tokenizes via the module-level helpers
``cosmos3_special_tokens`` / ``cosmos3_tokenize_caption``
(``fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline``), which wrap a Qwen2
chat-template tokenizer:

  * special tokens: ``start_of_generation=<|vision_start|>``,
    ``end_of_generation=<|vision_end|>``, ``eos_token_id=tokenizer.eos_token_id``
    (the framework ``llm_special_tokens``);
  * ``tokenize_caption`` applies the chat template with
    ``add_generation_prompt=True`` / ``add_vision_id=False`` and an optional
    image/video system prompt.

These contract checks run against the conftest Qwen2-shaped stub tokenizer (no
real weights). The byte-for-byte real-token-id check (``eos == 151645``,
``<|vision_start|> == 151652``) needs the real ``nvidia/Cosmos3-Nano``
``text_tokenizer`` and is skipped cleanly when it is unavailable.
"""
from __future__ import annotations

import pytest

from .conftest import StubQwen2Tokenizer

pytestmark = [pytest.mark.local]


def test_native_special_tokens_resolution() -> None:
    """``cosmos3_special_tokens`` resolves the three generation special tokens."""
    from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import cosmos3_special_tokens

    special = cosmos3_special_tokens(StubQwen2Tokenizer())
    assert set(special) == {"start_of_generation", "end_of_generation", "eos_token_id"}
    assert special["start_of_generation"] == StubQwen2Tokenizer().convert_tokens_to_ids("<|vision_start|>")
    assert special["end_of_generation"] == StubQwen2Tokenizer().convert_tokens_to_ids("<|vision_end|>")
    assert special["eos_token_id"] == StubQwen2Tokenizer.eos_token_id


def test_native_tokenize_caption_uses_chat_template() -> None:
    """``cosmos3_tokenize_caption`` returns a non-empty token-id list.

    The video / image system-prompt variants tokenize independently (the chat
    template prepends a role=system turn when ``use_system_prompt`` is set), and
    the result is always a plain list of ints.
    """
    from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import cosmos3_tokenize_caption

    tok = StubQwen2Tokenizer()
    ids_video = cosmos3_tokenize_caption(tok, "a robot dances", is_video=True, use_system_prompt=False)
    ids_image = cosmos3_tokenize_caption(tok, "a robot", is_video=False, use_system_prompt=True)
    assert isinstance(ids_video, list) and all(isinstance(i, int) for i in ids_video) and ids_video
    assert isinstance(ids_image, list) and ids_image


def test_cosmos3_special_token_ids_real_weights() -> None:
    """Byte-for-byte Qwen2 special-token ids (needs the real text_tokenizer).

    Asserts ``eos_token_id == 151645`` and
    ``convert_tokens_to_ids('<|vision_start|>') == 151652`` on the real
    ``nvidia/Cosmos3-Nano`` Qwen2 tokenizer. Skipped cleanly when the real
    tokenizer is not loadable in this environment.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not available")

    import os

    candidate_paths = [
        os.path.join(os.environ.get("COSMOS3_WEIGHTS_DIR", ""), "text_tokenizer"),
        "official_weights/cosmos3/text_tokenizer",
    ]
    tok_path = next((p for p in candidate_paths if p and os.path.isdir(p)), None)
    if tok_path is None:
        pytest.skip("real nvidia/Cosmos3-Nano text_tokenizer not available "
                    "(set COSMOS3_WEIGHTS_DIR or provide official_weights/cosmos3)")

    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    assert tokenizer.eos_token_id == 151645
    assert tokenizer.convert_tokens_to_ids("<|vision_start|>") == 151652
