# SPDX-License-Identifier: Apache-2.0
"""Open-H3-Regenerate-2K draft bundles: latent + prompt conditioning handed over by a paired draft worker."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY
from fastvideo.pipelines.basic.open_h3_regenerate_2k.draft_bundle import (
    BUNDLE_EXTRA_KEY,
    bundle_prompt_conditioning,
    load_draft_bundle,
)
from fastvideo.pipelines.basic.open_h3_regenerate_2k.stages.conditioning import Regenerate2KConditioningStage


def _bundle(with_prompt: bool = True) -> dict:
    bundle = {"format": 1, "latent": torch.randn(24, 3, 4, 6), "prompt": "a fox"}
    if with_prompt:
        bundle["prompt_embeds"] = torch.randn(7, 5120)
        bundle["text_token_tags"] = torch.ones(7, dtype=torch.long)
    return bundle


def _batch(path: str) -> SimpleNamespace:
    return SimpleNamespace(video_path=path, prompt="a fox", prompt_embeds=None, extra={})


def test_bundle_round_trip_and_bare_latent(tmp_path):
    path = tmp_path / "clip.pt"
    torch.save(_bundle(), path)
    bundle = load_draft_bundle(str(path))
    assert bundle is not None and bundle["latent"].shape == (24, 3, 4, 6)
    assert bundle_prompt_conditioning(bundle) is not None
    bare = tmp_path / "bare.pt"
    torch.save(torch.randn(24, 3, 4, 6), bare)
    assert load_draft_bundle(str(bare)) is None


def test_bundle_with_wrong_format_is_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format": 99, "latent": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="draft bundle"):
        load_draft_bundle(str(path))


@torch.no_grad()
def test_bundle_prompt_embeds_skip_the_text_encoder(tmp_path):
    path = tmp_path / "clip.pt"
    torch.save(_bundle(), path)
    # conditioner=None: touching the text encoder would fail, so this proves the encoder is never used.
    stage = Regenerate2KConditioningStage(conditioner=None, tokenizer=None, processor=None, ref2va=False)
    called = []
    stage.before_encode = lambda: called.append(True)
    batch = stage.forward(_batch(str(path)), SimpleNamespace(model_path="m"))
    assert batch.prompt_embeds[0].shape == (7, 5120)
    assert torch.equal(batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY], torch.ones(7, dtype=torch.long))
    assert batch.extra[BUNDLE_EXTRA_KEY]["latent"].shape == (24, 3, 4, 6)
    assert not called


@torch.no_grad()
def test_bundle_without_prompt_runs_the_release_hook_before_encoding(tmp_path):
    path = tmp_path / "clip.pt"
    torch.save(_bundle(with_prompt=False), path)
    stage = Regenerate2KConditioningStage(conditioner=None, tokenizer=None, processor=None, ref2va=False)
    called = []
    stage.before_encode = lambda: called.append(True)
    with pytest.raises(Exception):
        stage.forward(_batch(str(path)), SimpleNamespace(model_path="m"))  # no encoder in this test
    assert called


def test_latent_preparation_reads_the_bundle_latent(tmp_path):
    from fastvideo.pipelines.basic.open_h3_regenerate_2k.stages.latent_preparation import (
        Regenerate2KLatentPreparationStage)
    path = tmp_path / "clip.pt"
    bundle = _bundle()
    torch.save(bundle, path)
    stage = Regenerate2KLatentPreparationStage.__new__(Regenerate2KLatentPreparationStage)
    loaded = stage._load_draft_latent(str(path), torch.device("cpu"))
    assert torch.equal(loaded, bundle["latent"])
    reused = stage._load_draft_latent(str(path), torch.device("cpu"), {"latent": torch.zeros(24, 3, 4, 6)})
    assert torch.count_nonzero(reused) == 0
