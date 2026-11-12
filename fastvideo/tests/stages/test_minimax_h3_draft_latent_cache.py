# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the exact Open-H3-Regenerate-2K draft-latent cache."""

from types import SimpleNamespace

import torch

import fastvideo.envs as envs
from fastvideo.pipelines.basic.open_h3_regenerate_2k.draft_latent_cache import (
    DraftLatentCache,
    draft_latent_cache_key,
)
from fastvideo.pipelines.basic.open_h3_regenerate_2k.stages.latent_preparation import (
    Regenerate2KLatentPreparationStage,
)


def test_key_is_content_keyed(tmp_path) -> None:
    video = tmp_path / "draft.mp4"
    video.write_bytes(b"abc")
    base = draft_latent_cache_key(str(video), "768,768", None, "m", 5)
    assert base == draft_latent_cache_key(str(video), "768,768", None, "m", 5)
    assert base != draft_latent_cache_key(str(video), "768,768", None, "m", 6)
    assert base != draft_latent_cache_key(str(video), "512,512", None, "m", 5)
    assert base != draft_latent_cache_key(str(video), "768,768", None, "m2", 5)
    assert base != draft_latent_cache_key(str(video), "768,768", "768,768,128,128", "m", 5)
    video.write_bytes(b"abcd")  # different file content -> different key
    assert base != draft_latent_cache_key(str(video), "768,768", None, "m", 5)
    assert draft_latent_cache_key(str(tmp_path / "missing.mp4"), "768,768", None, "m", 5) is None
    assert draft_latent_cache_key(None, "768,768", None, "m", 5) is None


def test_cache_round_trip_clones_lru_and_disk(tmp_path) -> None:
    with envs.FASTVIDEO_H3_DRAFT_LATENT_CACHE.override(str(tmp_path)):
        cache = DraftLatentCache(capacity=1)
        assert cache.enabled
        draft = torch.randn(2, 3, 4)
        cache.put("a", draft)
        got = cache.get("a")
        assert got is not None and torch.equal(got, draft)
        got.add_(100)  # returned entry must be an independent clone
        assert torch.equal(cache.get("a"), draft)
        disk = DraftLatentCache(capacity=1).get("a")  # a fresh cache reads the disk entry
        assert disk is not None and torch.equal(disk, draft)
        cache.put("b", torch.zeros(5))
        assert cache.get("a") is not None  # evicted from the LRU, still served from disk
        assert cache.get("never") is None


def test_off_disables_the_cache(tmp_path) -> None:
    with envs.FASTVIDEO_H3_DRAFT_LATENT_CACHE.override("off"):
        cache = DraftLatentCache()
        assert not cache.enabled
        cache.put("a", torch.zeros(1))
        assert cache.get("a") is None


def test_cached_encode_skips_the_second_encode(tmp_path) -> None:
    video = tmp_path / "draft.mp4"
    video.write_bytes(b"pixels")
    with envs.FASTVIDEO_H3_DRAFT_LATENT_CACHE.override(str(tmp_path / "cache")):
        stage = Regenerate2KLatentPreparationStage(vae=None,
                                                   audio_vae=None,
                                                   scheduler=None,
                                                   load_upscaler=lambda: None,
                                                   draft_size="768,768",
                                                   start_sigma=None)
        calls = {"n": 0}

        def fake_encode(video_path, device, fastvideo_args):  # noqa: ANN001
            calls["n"] += 1
            return torch.full((2, 3), float(calls["n"]))

        stage._encode_draft = fake_encode
        args = SimpleNamespace(model_path="m", vae_cpu_offload=False, pin_cpu_memory=False)
        first = stage._encode_draft_cached(str(video), torch.device("cpu"), args)
        second = stage._encode_draft_cached(str(video), torch.device("cpu"), args)
    assert calls["n"] == 1
    assert torch.equal(first, second)
