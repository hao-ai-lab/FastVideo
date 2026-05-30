# SPDX-License-Identifier: Apache-2.0
"""Parity scaffold for GLM-Image glyph T5 encoder.

The text_encoder is a small byte-level T5 (d_model=1472, gated-gelu, ByT5
tokenizer, vocab_size=384). This test exercises the glyph-only encoding path
that the diffusers pipeline implements as `_get_glyph_embeds` and compares
against the FastVideo glyph-encoding stage logic. Until the FastVideo stage is
refactored to mirror diffusers (per-prompt loop, even-length pad-prefix,
attention-mask flattening, left-padded batch), the parity test asserts the
high-impact pieces individually so failures point to the exact path that
diverged.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")
os.environ.setdefault("DISABLE_SP", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY = "glm_image"
LOCAL_WEIGHTS_DIR = Path(
    os.getenv("GLM_IMAGE_LOCAL_WEIGHTS_DIR",
              REPO_ROOT / "official_weights" / FAMILY))
TEXT_ENCODER_DIR = LOCAL_WEIGHTS_DIR / "text_encoder"
TOKENIZER_DIR = LOCAL_WEIGHTS_DIR / "tokenizer"


def _has_weights() -> bool:
    return (TEXT_ENCODER_DIR / "model.safetensors").exists() and TOKENIZER_DIR.exists()


pytestmark = pytest.mark.skipif(
    not _has_weights(),
    reason=f"GLM-Image text_encoder/tokenizer not found under {LOCAL_WEIGHTS_DIR}.",
)


SAMPLE_PROMPT = (
    'A photo of a coffee shop with a wooden sign reading "Daily Grind" '
    "and a chalk menu next to it"
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GLM-Image glyph encoder parity.")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def official_tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.ByT5Tokenizer.from_pretrained(str(TOKENIZER_DIR))


@pytest.fixture(scope="module")
def official_text_encoder(device):
    transformers = pytest.importorskip("transformers")
    return transformers.T5EncoderModel.from_pretrained(
        str(TEXT_ENCODER_DIR), torch_dtype=torch.float32).to(device).eval()


def test_tokenizer_is_byt5(official_tokenizer):
    """Glyph tokenizer must be ByT5 (byte-level), not generic T5."""
    assert type(official_tokenizer).__name__ == "ByT5Tokenizer"
    assert official_tokenizer.vocab_size <= 400, (
        "ByT5 vocab is byte-level (~256+special)")


def test_glyph_text_extraction():
    """Glyph extraction must pull single+double+CJK quoted spans."""
    pytest.importorskip("fastvideo")
    try:
        from fastvideo.pipelines.basic.glm_image.stages.before_denoising import (  # noqa
            get_glyph_texts)
    except ImportError as e:
        pytest.skip(f"FastVideo glyph extractor not yet at target path: {e}")
    out = get_glyph_texts(SAMPLE_PROMPT)
    assert "Daily Grind" in out


def _diffusers_reference_get_glyph_embeds(prompts, tokenizer, text_encoder,
                                          device, dtype,
                                          max_sequence_length=2048):
    """Inlined copy of diffusers `_get_glyph_embeds` + `get_glyph_texts`.

    Source: `reference/diffusers/src/diffusers/pipelines/glm_image/`
    `pipeline_glm_image.py::GlmImagePipeline._get_glyph_embeds` at commit
    ff3b86b4755b46a7b5656dfcf84d25bd25ad4740 (diffusers main, 0.37.0.dev0).
    Inlined here because the installed diffusers package (`__version__`
    typically <0.37.0) does not yet ship `GlmImagePipeline`. This function is
    the reference oracle for FastVideo's `compute_glyph_embeds`.
    """
    import re

    def get_glyph_texts(prompt):
        if isinstance(prompt, str):
            prompt = [prompt]
        out = []
        for p in prompt:
            out.append(
                re.findall(r"'([^']*)'", p)
                + re.findall(r"“([^“”]*)”", p)
                + re.findall(r'"([^"]*)"', p)
                + re.findall(r"「([^「」]*)」", p))
        return out

    all_glyph_texts = get_glyph_texts(prompts)
    all_glyph_embeds = []
    for glyph_texts in all_glyph_texts:
        if len(glyph_texts) == 0:
            glyph_texts = [""]
        input_ids = tokenizer(
            glyph_texts,
            max_length=max_sequence_length,
            truncation=True,
        ).input_ids
        input_ids = [
            [tokenizer.pad_token_id] * ((len(input_ids) + 1) % 2) + ids
            for ids in input_ids
        ]
        max_length = max(len(ids) for ids in input_ids)
        attention_mask = torch.tensor(
            [[1] * len(ids) + [0] * (max_length - len(ids))
             for ids in input_ids],
            device=device,
        )
        input_ids_t = torch.tensor(
            [
                ids + [tokenizer.pad_token_id] * (max_length - len(ids))
                for ids in input_ids
            ],
            device=device,
        )
        outputs = text_encoder(input_ids_t, attention_mask=attention_mask)
        glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)
        all_glyph_embeds.append(glyph_embeds)

    max_seq_len = max(emb.size(1) for emb in all_glyph_embeds)
    padded = []
    for emb in all_glyph_embeds:
        if emb.size(1) < max_seq_len:
            pad = torch.zeros(emb.size(0), max_seq_len - emb.size(1),
                              emb.size(2), device=device, dtype=emb.dtype)
            emb = torch.cat([pad, emb], dim=1)
        padded.append(emb)
    return torch.cat(padded, dim=0).to(device=device, dtype=dtype)


def test_get_glyph_embeds_matches_diffusers(official_tokenizer,
                                            official_text_encoder, device):
    """FastVideo `compute_glyph_embeds` must equal the diffusers reference."""
    pytest.importorskip("fastvideo")
    try:
        from fastvideo.pipelines.basic.glm_image.stages.before_denoising import (  # noqa
            compute_glyph_embeds)
    except ImportError as e:
        pytest.skip(f"FastVideo glyph encoder not yet at target path: {e}")

    with torch.no_grad():
        official_embeds = _diffusers_reference_get_glyph_embeds(
            [SAMPLE_PROMPT], official_tokenizer, official_text_encoder, device,
            torch.float32)
        fv_embeds = compute_glyph_embeds([SAMPLE_PROMPT],
                                         tokenizer=official_tokenizer,
                                         text_encoder=official_text_encoder,
                                         device=device,
                                         dtype=torch.float32,
                                         max_sequence_length=2048)
    assert fv_embeds.shape == official_embeds.shape, (
        f"shape mismatch: {fv_embeds.shape} vs {official_embeds.shape}")
    torch.testing.assert_close(fv_embeds, official_embeds, atol=1e-4,
                               rtol=1e-4)
