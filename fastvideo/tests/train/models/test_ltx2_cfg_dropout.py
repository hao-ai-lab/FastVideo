# SPDX-License-Identifier: Apache-2.0
"""CFG dropout on LTX-2 swaps in the unconditional embedding, not zeros.

LTX-2 builds its unconditional branch from an empty prompt run through
Gemma and the Embeddings1D connector.  The shared parquet path drops text
conditioning by zeroing the stored embedding, which would train against a
different unconditional than the sampler uses, so ``LTX2Model`` performs
the drop itself.  These tests cover the substitution in isolation; they do
not load the 22B transformer or Gemma.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29527")

import torch

from fastvideo.train.models.ltx2.ltx2 import LTX2Model

BATCH, TOKENS, DIM = 4, 8, 16


def _model(cfg_rate: float) -> LTX2Model:
    """An LTX2Model shell carrying only what ``_apply_cfg_dropout`` reads."""
    model = LTX2Model.__new__(LTX2Model)
    model._training_cfg_rate = cfg_rate
    model.negative_prompt_embeds = torch.full((1, TOKENS, DIM), 7.0)
    model.negative_prompt_attention_mask = torch.ones((1, TOKENS))
    # The cache is already populated, so no encoder needs loading.
    model.ensure_negative_conditioning = lambda: None  # type: ignore[method-assign]
    return model


def _conditioning() -> tuple[torch.Tensor, torch.Tensor]:
    embeds = torch.arange(BATCH * TOKENS * DIM, dtype=torch.float32)
    return embeds.view(BATCH, TOKENS, DIM), torch.ones((BATCH, TOKENS))


def test_cfg_rate_zero_is_a_no_op() -> None:
    model = _model(0.0)
    embeds, mask = _conditioning()
    out_embeds, out_mask = model._apply_cfg_dropout(
        embeds, mask, generator=torch.Generator().manual_seed(0))
    assert out_embeds is embeds
    assert out_mask is mask


def test_dropped_samples_carry_the_unconditional_embedding() -> None:
    model = _model(1.0)  # drop every sample
    embeds, mask = _conditioning()
    out_embeds, out_mask = model._apply_cfg_dropout(
        embeds, mask, generator=torch.Generator().manual_seed(0))

    expected = model.negative_prompt_embeds[0]
    for i in range(BATCH):
        assert torch.equal(out_embeds[i], expected)
        # The pre-fix behaviour zeroed the embedding; make sure we do not.
        assert not torch.equal(out_embeds[i], torch.zeros_like(out_embeds[i]))
    assert torch.equal(out_mask, model.negative_prompt_attention_mask.expand(BATCH, TOKENS))


def test_kept_samples_are_untouched() -> None:
    model = _model(1.0)
    embeds, mask = _conditioning()
    original = embeds.clone()
    model._apply_cfg_dropout(embeds, mask, generator=torch.Generator().manual_seed(0))
    # The input tensors are cloned before substitution.
    assert torch.equal(embeds, original)
