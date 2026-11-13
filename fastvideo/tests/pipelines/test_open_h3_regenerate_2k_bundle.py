# SPDX-License-Identifier: Apache-2.0
"""Open-H3-Regenerate-2K draft bundles: latent + prompt conditioning handed over by a paired draft worker."""

import os
from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY
from fastvideo.pipelines.basic.minimax_h3.draft_bundle import (
    BUNDLE_EXTRA_KEY,
    bundle_prompt_conditioning,
    BUNDLE_PATH_EXTRA_KEY,
    BUNDLE_TEXT_TOKEN_TAGS_EXTRA_KEY,
    keep_text_token_tags_for_bundle,
    load_draft_bundle,
    save_draft_bundle,
    text_encoder_fingerprint,
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
    stage.prompt_embeds_policy = "always"
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


def test_prompt_embeds_policy(tmp_path):
    bundle = _bundle()
    bundle["text_encoder_fingerprint"] = "abc"
    assert bundle_prompt_conditioning(bundle, policy="always") is not None
    assert bundle_prompt_conditioning(bundle, policy="never") is None
    assert bundle_prompt_conditioning(bundle, policy="match", text_encoder="abc") is not None
    assert bundle_prompt_conditioning(bundle, policy="match", text_encoder="other") is None
    assert bundle_prompt_conditioning(bundle, policy="match", text_encoder=None) is None
    with pytest.raises(ValueError, match="policy"):
        bundle_prompt_conditioning(bundle, policy="sometimes")


def test_prompt_mismatch_re_encodes_under_match():
    bundle = _bundle()
    bundle["text_encoder_fingerprint"] = "abc"
    assert bundle_prompt_conditioning(bundle, policy="match", text_encoder="abc", prompt="a fox") is not None
    assert bundle_prompt_conditioning(bundle, policy="match", text_encoder="abc", prompt="a wolf") is None
    assert bundle_prompt_conditioning(bundle, policy="always", text_encoder="abc", prompt="a wolf") is not None


def test_prompt_embeds_policies_match_the_pipeline_config():
    from fastvideo.configs.pipelines.open_h3_regenerate_2k import PROMPT_EMBEDS_POLICIES as config_policies
    from fastvideo.pipelines.basic.minimax_h3.draft_bundle import PROMPT_EMBEDS_POLICIES
    assert config_policies == PROMPT_EMBEDS_POLICIES


@torch.no_grad()
def test_mismatched_encoder_re_encodes_after_the_release_hook(tmp_path):
    path = tmp_path / "clip.pt"
    bundle = _bundle()
    bundle["text_encoder_fingerprint"] = "theirs"
    torch.save(bundle, path)
    stage = Regenerate2KConditioningStage(conditioner=None, tokenizer=None, processor=None, ref2va=False)
    stage.text_encoder_fingerprint = lambda: "ours"
    called = []
    stage.before_encode = lambda: called.append(True)
    with pytest.raises(Exception):
        stage.forward(_batch(str(path)), SimpleNamespace(model_path="m"))  # falls through to the (absent) encoder
    assert called


def test_text_encoder_fingerprint_tracks_files(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for root in (first, second):
        root.mkdir()
        (root / "config.json").write_text('{"hidden_size": 5120}')
        (root / "model-00001-of-00002.safetensors").write_bytes(b"x" * 10)
    assert text_encoder_fingerprint(first) == text_encoder_fingerprint(second)
    assert text_encoder_fingerprint(first) != text_encoder_fingerprint(first, quant="nvfp4")
    (second / "model-00002-of-00002.safetensors").write_bytes(b"y")
    assert text_encoder_fingerprint(first) != text_encoder_fingerprint(second)
    assert text_encoder_fingerprint(tmp_path / "missing") is None


def test_text_encoder_fingerprint_uses_only_the_selected_file(tmp_path):
    folder = tmp_path / "enc"
    folder.mkdir()
    selected = folder / "encoder.safetensors"
    selected.write_bytes(b"aaaa")
    before = text_encoder_fingerprint(selected)
    (folder / "sibling.safetensors").write_bytes(b"bbbb")
    assert text_encoder_fingerprint(selected) == before  # a sibling is not part of a single-file fingerprint

    twin_dir = tmp_path / "twin"
    twin_dir.mkdir()
    twin = twin_dir / "encoder.safetensors"
    twin.write_bytes(b"aaaa")
    assert text_encoder_fingerprint(twin) == before  # same bytes and name match across folders
    twin.write_bytes(b"cccc")
    os.utime(twin, ns=(2, 2))  # force a fresh stat key for the content cache
    assert text_encoder_fingerprint(twin) != before  # same size, different bytes


def test_save_then_load_round_trip(tmp_path):
    target = tmp_path / "out" / "clip.h3draft.pt"
    latent = torch.randn(1, 24, 3, 4, 6)
    save_draft_bundle(target, latent=latent, prompt_embeds=torch.ones(1, 7, 5120),
                      text_token_tags=torch.ones(7, dtype=torch.long), prompt="a fox", text_encoder="fp")
    bundle = load_draft_bundle(str(target))
    assert torch.equal(bundle["latent"], latent[0]) and bundle["text_encoder_fingerprint"] == "fp"
    assert bundle["prompt"] == "a fox" and not list(target.parent.glob(".*.tmp"))
    with pytest.raises(ValueError, match="C, T, h, w"):
        save_draft_bundle(target, latent=torch.zeros(3, 3), prompt_embeds=None, text_token_tags=None, prompt="",
                          text_encoder=None)


def test_text_token_tags_survive_latent_preparation_only_for_bundle_requests():
    tags = torch.ones(5, dtype=torch.long)
    plain: dict = {}
    keep_text_token_tags_for_bundle(plain, tags)
    assert BUNDLE_TEXT_TOKEN_TAGS_EXTRA_KEY not in plain
    requested = {BUNDLE_PATH_EXTRA_KEY: "/tmp/x.h3draft.pt"}
    keep_text_token_tags_for_bundle(requested, tags)
    assert requested[BUNDLE_TEXT_TOKEN_TAGS_EXTRA_KEY] is tags
    keep_text_token_tags_for_bundle(requested, None)
    assert requested[BUNDLE_TEXT_TOKEN_TAGS_EXTRA_KEY] is tags
