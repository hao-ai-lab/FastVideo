# SPDX-License-Identifier: Apache-2.0
"""Regression test for ``kandinsky6_qwen_preprocess_text`` against the diffusers reference
(``pipeline_kandinsky6_ti2va.py``): an empty prompt encodes an empty user turn. The old code substituted
``"."`` for an empty prompt, which tokenizes differently from the reference's empty user turn (the
template's system half is never empty, so the Qwen tokenizer never sees zero tokens either way -- the
"." guard was unnecessary here, unlike the generic `treat_empty_as_dot` escape hatch other encoders use
for tokenizers that genuinely produce zero tokens on "").

Pure string test -- no GPU, no model weights, no tokenizer.
"""
from __future__ import annotations

from fastvideo.configs.pipelines.kandinsky6 import (
    KANDINSKY6_PROMPT_TEMPLATE,
    kandinsky6_qwen_preprocess_text,
)


def test_empty_prompt_encodes_an_empty_user_turn_not_a_dot():
    result = kandinsky6_qwen_preprocess_text("")

    assert result == KANDINSKY6_PROMPT_TEMPLATE.format("")
    assert "user\n<|im_end|>" in result
    assert "user\n.<|im_end|>" not in result


def test_whitespace_only_prompt_is_also_left_as_is():
    result = kandinsky6_qwen_preprocess_text("   ")

    assert result == KANDINSKY6_PROMPT_TEMPLATE.format("   ")


def test_non_empty_prompt_is_unaffected():
    result = kandinsky6_qwen_preprocess_text("a chef chops vegetables")

    assert result == KANDINSKY6_PROMPT_TEMPLATE.format("a chef chops vegetables")
