# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the content-keyed MiniMax-H3 reference-encode memo."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import fastvideo.envs as envs
from fastvideo.pipelines.basic.minimax_h3.memo import ContentMemo, image_key
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import (
    MiniMaxH3LatentPreparationStage, )


def _image(seed: int, size: tuple[int, int] = (8, 6)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


@torch.no_grad()
def test_memo_returns_independent_clones_and_evicts_lru() -> None:
    memo = ContentMemo(capacity=2)
    calls = []

    def compute(value):
        calls.append(value)
        return torch.full((3, ), float(value)), torch.tensor([value])

    first = memo.get_or_compute("a", lambda: compute(1))
    first[0].add_(100)
    again = memo.get_or_compute("a", lambda: compute(1))
    assert calls == [1]
    assert torch.equal(again[0], torch.full((3, ), 1.0))
    again[0].add_(100)
    assert torch.equal(memo.get_or_compute("a", lambda: compute(1))[0], torch.full((3, ), 1.0))

    memo.get_or_compute("b", lambda: compute(2))
    memo.get_or_compute("a", lambda: compute(1))
    memo.get_or_compute("c", lambda: compute(3))
    assert len(memo) == 2
    memo.get_or_compute("b", lambda: compute(2))
    assert calls == [1, 2, 3, 2]


def test_memo_is_off_by_default_and_under_autograd() -> None:
    with envs.FASTVIDEO_H3_REF2VA_MEMO_ENTRIES.override(None):
        assert not ContentMemo().enabled
    with envs.FASTVIDEO_H3_REF2VA_MEMO_ENTRIES.override(4):
        memo = ContentMemo()
        assert memo.capacity == 4
        with torch.enable_grad():
            assert not memo.enabled
            memo.get_or_compute("a", lambda: torch.zeros(1))
        assert len(memo) == 0


def test_image_key_is_content_keyed() -> None:
    assert image_key(_image(0)) == image_key(_image(0))
    assert image_key(_image(0)) != image_key(_image(1))
    pixels = np.asarray(_image(0)).copy()
    changed = pixels.copy()
    changed[0, 0, 0] ^= 1
    assert image_key(pixels) != image_key(changed)
    assert image_key(pixels) != image_key(pixels.astype(np.uint16))
    assert image_key(pixels) != image_key(pixels.reshape(-1, 3))


@pytest.fixture
def memo_entries():
    with envs.FASTVIDEO_H3_REF2VA_MEMO_ENTRIES.override(4):
        yield


def _conditioning_stage(monkeypatch):
    stage = MiniMaxH3ConditioningStage(conditioner=None, tokenizer=None, processor=None, ref2va=True)
    calls = []

    def present(batch, references, device):
        calls.append(batch.prompt)
        return torch.randn(1, 5, 4), torch.zeros(5, dtype=torch.long)

    monkeypatch.setattr(stage, "_present_ref2va", present)
    return stage, calls


@torch.no_grad()
def test_ref2va_presentation_is_reused_for_identical_content(monkeypatch, memo_entries) -> None:
    stage, calls = _conditioning_stage(monkeypatch)
    device = torch.device("cpu")

    def batch(prompt, *seeds, audio=False):
        references = [MiniMaxH3PreparedReference("image", image=_image(seed)) for seed in seeds]
        if audio:
            references.append(MiniMaxH3PreparedReference("audio", has_audio=True))
        return SimpleNamespace(prompt=prompt, references=references)

    first = stage._encode_ref2va(batch("p", 0, 1), device)
    repeat = stage._encode_ref2va(batch("p", 0, 1), device)
    assert len(calls) == 1
    assert all(torch.equal(a, b) for a, b in zip(first, repeat, strict=True))

    stage._encode_ref2va(batch("q", 0, 1), device)
    stage._encode_ref2va(batch("p", 1, 0), device)
    stage._encode_ref2va(batch("p", 0, 2), device)
    stage._encode_ref2va(batch("p", 0, 1, audio=True), device)
    assert len(calls) == 5


@torch.no_grad()
def test_ref2va_with_a_video_reference_is_never_memoized(monkeypatch, memo_entries) -> None:
    stage, calls = _conditioning_stage(monkeypatch)
    references = [
        MiniMaxH3PreparedReference("image", image=_image(0)),
        MiniMaxH3PreparedReference("video", frames=np.zeros((4, 6, 8, 3), dtype=np.uint8)),
    ]
    for _ in range(2):
        stage._encode_ref2va(SimpleNamespace(prompt="p", references=references), torch.device("cpu"))
    assert len(calls) == 2


@torch.no_grad()
def test_keyframe_latents_are_reused_for_identical_pixels(monkeypatch, memo_entries) -> None:
    stage = MiniMaxH3LatentPreparationStage(vae=None, audio_vae=None, scheduler=None, ref2va=True)
    calls = []

    def encode(image, device):
        calls.append(image_key(image))
        return torch.randn(1, 4, 1, 3, 2)

    monkeypatch.setattr(stage, "_encode_keyframe_latents_uncached", encode)
    device = torch.device("cpu")
    first = stage._encode_keyframe_latents(_image(0), device)
    assert torch.equal(stage._encode_keyframe_latents(_image(0), device), first)
    stage._encode_keyframe_latents(_image(1), device)
    assert len(calls) == 2


@pytest.mark.parametrize("entries", (0, None))
@torch.no_grad()
def test_stages_recompute_without_the_flag(monkeypatch, entries) -> None:
    with envs.FASTVIDEO_H3_REF2VA_MEMO_ENTRIES.override(entries):
        stage = MiniMaxH3LatentPreparationStage(vae=None, audio_vae=None, scheduler=None, ref2va=True)
        calls = []
        monkeypatch.setattr(stage, "_encode_keyframe_latents_uncached",
                            lambda image, device: calls.append(1) or torch.zeros(1))
        for _ in range(3):
            stage._encode_keyframe_latents(_image(0), torch.device("cpu"))
        assert len(calls) == 3
