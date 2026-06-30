# SPDX-License-Identifier: Apache-2.0
"""
Parity test for Z-Image tokenizer loading and tokenization behavior.

This compares:
1) direct transformers AutoTokenizer loading from local tokenizer dir, and
2) FastVideo TokenizerLoader loading path,

using identical prompts and tokenization settings.

Usage:
    pytest tests/local_tests/zimage/test_zimage_tokenizer_parity.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close
from transformers import AutoTokenizer

from fastvideo.models.loader.component_loader import TokenizerLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
ZIMAGE_TOKENIZER_DIR = REPO_ROOT / "official_weights" / "Z-Image" / "tokenizer"


@dataclass
class _DummyPipelineConfig:
    text_encoder_configs: tuple = field(default_factory=tuple)


@dataclass
class _DummyFastVideoArgs:
    pipeline_config: _DummyPipelineConfig = field(default_factory=_DummyPipelineConfig)


def _load_reference_tokenizer():
    if not ZIMAGE_TOKENIZER_DIR.exists():
        pytest.skip(f"Z-Image tokenizer dir not found: {ZIMAGE_TOKENIZER_DIR}")
    return AutoTokenizer.from_pretrained(str(ZIMAGE_TOKENIZER_DIR), local_files_only=True)


def _load_fastvideo_tokenizer():
    if not ZIMAGE_TOKENIZER_DIR.exists():
        pytest.skip(f"Z-Image tokenizer dir not found: {ZIMAGE_TOKENIZER_DIR}")
    loader = TokenizerLoader()
    return loader.load(str(ZIMAGE_TOKENIZER_DIR), _DummyFastVideoArgs())


def test_zimage_tokenizer_loader_and_tokenization_parity():
    ref_tok = _load_reference_tokenizer()
    fv_tok = _load_fastvideo_tokenizer()

    prompts = [
        "A close-up portrait of a cat in warm studio lighting.",
        "An oil painting of a lighthouse at sunset.",
    ]

    tok_kwargs = {
        "padding": "max_length",
        "max_length": 96,
        "truncation": True,
        "return_tensors": "pt",
    }

    ref_out = ref_tok(prompts, **tok_kwargs)
    fv_out = fv_tok(prompts, **tok_kwargs)

    assert_close(ref_out["input_ids"], fv_out["input_ids"], atol=0, rtol=0)
    assert_close(ref_out["attention_mask"], fv_out["attention_mask"], atol=0, rtol=0)

    # Compare key tokenizer attributes that affect generation-time behavior.
    assert ref_tok.padding_side == fv_tok.padding_side
    assert ref_tok.pad_token_id == fv_tok.pad_token_id
    assert ref_tok.eos_token_id == fv_tok.eos_token_id


@pytest.mark.skipif(not ZIMAGE_TOKENIZER_DIR.exists(), reason="Z-Image tokenizer assets are required")
def test_zimage_tokenizer_chat_template_parity():
    ref_tok = _load_reference_tokenizer()
    fv_tok = _load_fastvideo_tokenizer()

    if not hasattr(ref_tok, "apply_chat_template") or not hasattr(fv_tok, "apply_chat_template"):
        pytest.skip("Tokenizer does not expose apply_chat_template")

    messages = [{"role": "user", "content": "Describe a futuristic city skyline."}]

    # Keep kwargs minimal for broad compatibility across tokenizer versions.
    ref_text = ref_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    fv_text = fv_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    assert ref_text == fv_text

    ref_tokens = ref_tok(ref_text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    fv_tokens = fv_tok(fv_text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

    assert_close(ref_tokens["input_ids"], fv_tokens["input_ids"], atol=0, rtol=0)
    assert_close(ref_tokens["attention_mask"], fv_tokens["attention_mask"], atol=0, rtol=0)
