# SPDX-License-Identifier: Apache-2.0
"""CFG dropout on LTX-2 swaps in the unconditional embedding, not zeros.

LTX-2 builds its unconditional branch from the checkpoint preset's
``negative_prompt`` (empty for the distilled presets, the quality-negative
prompt for the base presets) run through Gemma and the Embeddings1D
connector.  The shared parquet path drops text conditioning by zeroing the
stored embedding, which would train against a different unconditional than
the sampler uses, so ``LTX2Model`` performs the drop itself and disables the
collate's zeroing drop.  These tests cover the substitution in isolation;
they do not load the 22B transformer or Gemma.
"""

from __future__ import annotations

import torch

import fastvideo.envs as envs

envs.setdefault_external("MASTER_ADDR", "localhost")
envs.setdefault_external("MASTER_PORT", "29527")

from fastvideo.train.models.ltx2.ltx2 import LTX2Model, _resolve_unconditional_prompt
from fastvideo.train.models.wan.wan import WanModel

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


def test_mixed_batch_drops_only_the_drawn_rows() -> None:
    """In a partially-dropped batch, only the drawn rows are substituted."""
    model = _model(0.5)
    embeds, mask = _conditioning()
    original = embeds.clone()
    expected_neg = model.negative_prompt_embeds[0]

    for seed in range(32):
        out_embeds, out_mask = model._apply_cfg_dropout(
            embeds, mask, generator=torch.Generator().manual_seed(seed))
        # A twin generator replays the draw to learn which rows dropped.
        keep = torch.rand(BATCH, generator=torch.Generator().manual_seed(seed))
        dropped = (keep < 0.5).tolist()
        if not (any(dropped) and any(not d for d in dropped)):
            continue  # this test needs a mixed drop/keep batch
        for i, is_dropped in enumerate(dropped):
            if is_dropped:
                assert torch.equal(out_embeds[i], expected_neg)
                assert torch.equal(
                    out_mask[i], model.negative_prompt_attention_mask[0])
            else:
                assert torch.equal(out_embeds[i], original[i])
                assert torch.equal(out_mask[i], mask[i])
        # The input tensors are cloned before substitution.
        assert torch.equal(embeds, original)
        return
    raise AssertionError("no mixed drop/keep batch found in 32 seeds")


def test_unconditional_length_mismatch_raises() -> None:
    """A text_len mismatch fails loudly instead of silent truncation."""
    model = _model(1.0)  # drop every sample so the substitution path runs
    model.negative_prompt_embeds = torch.full((1, TOKENS + 1, DIM), 7.0)
    model.negative_prompt_attention_mask = torch.ones((1, TOKENS + 1))
    embeds, mask = _conditioning()
    try:
        model._apply_cfg_dropout(
            embeds, mask, generator=torch.Generator().manual_seed(0))
    except ValueError as exc:
        assert "unconditional embedding length" in str(exc)
    else:
        raise AssertionError(
            "expected ValueError for mismatched unconditional length")


def test_ltx2_disables_the_dataloader_zeroing_drop() -> None:
    """The model-side drop must not stack with the collate's zeroing drop."""
    assert LTX2Model.__new__(LTX2Model)._dataloader_cfg_rate() == 0.0
    # WanModel keeps the shared zeroing drop (no model-side drop there).
    assert WanModel.__new__(WanModel)._dataloader_cfg_rate() is None


def test_unconditional_prompt_follows_the_checkpoint_preset() -> None:
    """Training drops to the prompt the sampler feeds its uncond branch."""
    # Distilled presets: negative_prompt="" (guidance 1.0).
    assert _resolve_unconditional_prompt(
        "FastVideo/LTX2-Distilled-Diffusers") == ""
    # Base presets run CFG against their quality-negative prompt.
    assert _resolve_unconditional_prompt("Lightricks/LTX-2").startswith("blurry")
