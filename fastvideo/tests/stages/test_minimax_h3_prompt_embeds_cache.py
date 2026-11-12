# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the exact MiniMax-H3 prompt-embeds cache."""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import fastvideo.envs as envs
from fastvideo.distributed import get_local_torch_device
from fastvideo.pipelines.basic.minimax_h3.prompt_embeds_cache import PromptEmbedsCache
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import (
    MiniMaxH3ConditioningStage,
    prompt_embeds_cache_key,
)


def _image(seed: int, size: tuple[int, int] = (8, 6)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


def _batch(prompt: str = "p", keyframes=(), anchors=(), references=()) -> SimpleNamespace:
    return SimpleNamespace(prompt=prompt,
                           extra={
                               "minimax_h3_keyframes": list(keyframes),
                               "minimax_h3_keyframe_anchors": tuple(anchors),
                           },
                           references=list(references))


def test_key_is_content_keyed_and_refuses_video_references() -> None:
    device = torch.device("cpu")
    base = prompt_embeds_cache_key(_batch("hello"), False, "m", device)
    assert base == prompt_embeds_cache_key(_batch("hello"), False, "m", device)
    assert base != prompt_embeds_cache_key(_batch("other"), False, "m", device)
    assert base != prompt_embeds_cache_key(_batch("hello"), True, "m", device)
    assert base != prompt_embeds_cache_key(_batch("hello"), False, "m2", device)
    assert base != prompt_embeds_cache_key(_batch("hello", keyframes=[_image(0)], anchors=["first"]), False, "m",
                                           device)
    video = [MiniMaxH3PreparedReference("video", frames=np.zeros((2, 4, 4, 3), dtype=np.uint8))]
    assert prompt_embeds_cache_key(_batch("p", references=video), True, "m", device) is None


def test_cache_round_trip_clones_lru_and_disk(tmp_path) -> None:
    with envs.FASTVIDEO_H3_PROMPT_EMBEDS_CACHE.override(str(tmp_path)):
        cache = PromptEmbedsCache(capacity=1)
        assert cache.enabled
        embeds = torch.randn(1, 4, 3)
        tags = torch.arange(4)
        cache.put("a", embeds, tags)
        got = cache.get("a")
        assert got is not None
        assert torch.equal(got[0], embeds) and torch.equal(got[1], tags)
        got[0].add_(100)  # returned entry must be an independent clone
        assert torch.equal(cache.get("a")[0], embeds)
        disk = PromptEmbedsCache(capacity=1).get("a")  # a fresh cache reads the disk entry
        assert disk is not None and torch.equal(disk[0], embeds)
        cache.put("b", torch.zeros(1, 2, 3), torch.zeros(2, dtype=torch.long))
        assert cache.get("a") is not None  # evicted from the LRU, still served from disk
        assert cache.get("never") is None


def test_off_disables_the_cache() -> None:
    with envs.FASTVIDEO_H3_PROMPT_EMBEDS_CACHE.override("off"):
        cache = PromptEmbedsCache()
        assert not cache.enabled
        cache.put("a", torch.zeros(1), torch.zeros(1, dtype=torch.long))
        assert cache.get("a") is None


@torch.no_grad()
def test_forward_hit_never_touches_the_conditioner() -> None:
    # conditioner=None makes any access a hard failure, so this asserts the hit path
    # returns before the text encoder is ever materialized.
    with envs.FASTVIDEO_H3_PROMPT_EMBEDS_CACHE.override("memory"):
        stage = MiniMaxH3ConditioningStage(conditioner=None, tokenizer=None, processor=None, ref2va=False)
        batch = _batch("p")
        key = prompt_embeds_cache_key(batch, False, "m", get_local_torch_device())
        embeds = torch.randn(1, 4, 3)
        tags = torch.arange(4)
        stage._prompt_embeds_cache.put(key, embeds, tags)
        out = stage.forward(batch, SimpleNamespace(model_path="m"))
    assert torch.equal(out.prompt_embeds[0].cpu(), embeds)
    assert torch.equal(out.extra["minimax_h3_text_token_tags"], tags)


def test_cache_is_off_unless_opted_in() -> None:
    with envs.FASTVIDEO_H3_PROMPT_EMBEDS_CACHE.override(None):
        assert not PromptEmbedsCache().enabled
    with envs.FASTVIDEO_H3_PROMPT_EMBEDS_CACHE.override("memory"):
        cache = PromptEmbedsCache()
        assert cache.enabled
        cache.put("a", torch.ones(2), torch.zeros(2, dtype=torch.long))
        assert torch.equal(cache.get("a")[0], torch.ones(2))
