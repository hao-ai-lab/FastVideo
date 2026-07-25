# SPDX-License-Identifier: Apache-2.0
"""Tests for the MMAudio-compatible precomputed feature cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fastvideo.dataset.mmaudio_feature_dataset import (
    MMAudioFeatureDataset,
    _expand_cache_root,
    compute_mmaudio_latent_stats,
)

tensordict = pytest.importorskip("tensordict")


def _write_cache(path: Path, *, with_video: bool) -> None:
    tensors = {
        "mean": torch.randn(2, 4, 3),
        "std": torch.rand(2, 4, 3),
        "text_features": torch.randn(2, 5, 7),
    }
    if with_video:
        tensors["clip_features"] = torch.randn(2, 6, 8)
        tensors["sync_features"] = torch.randn(2, 9, 10)
    tensordict.TensorDict(tensors, batch_size=[2]).memmap_(str(path))


def _load(path: Path, *, include_metadata: bool = False) -> MMAudioFeatureDataset:
    return MMAudioFeatureDataset(
        path,
        latent_seq_len=4,
        latent_dim=3,
        clip_seq_len=6,
        clip_dim=8,
        sync_seq_len=9,
        sync_dim=10,
        text_seq_len=5,
        text_dim=7,
        include_metadata=include_metadata,
    )


@pytest.mark.parametrize("with_video", [False, True])
def test_mmaudio_feature_cache_contract(
    tmp_path: Path,
    with_video: bool,
) -> None:
    cache = tmp_path / "cache"
    _write_cache(cache, with_video=with_video)

    dataset = _load(cache)
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["audio_latent_mean"].shape == (4, 3)
    assert sample["audio_latent_std"].shape == (4, 3)
    assert sample["clip_features"].shape == (6, 8)
    assert sample["sync_features"].shape == (9, 10)
    assert sample["text_features"].shape == (5, 7)
    assert sample["video_exists"].item() is with_video
    assert sample["text_exists"].item() is True
    if not with_video:
        assert torch.count_nonzero(sample["clip_features"]) == 0
        assert torch.count_nonzero(sample["sync_features"]) == 0


def test_mmaudio_feature_cache_rejects_wrong_shape(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _write_cache(cache, with_video=True)

    with pytest.raises(ValueError, match="mean.*per-sample shape"):
        MMAudioFeatureDataset(
            cache,
            latent_seq_len=99,
            latent_dim=3,
            clip_seq_len=6,
            clip_dim=8,
            sync_seq_len=9,
            sync_dim=10,
            text_seq_len=5,
            text_dim=7,
        )


def test_mmaudio_feature_cache_loads_validation_metadata(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _write_cache(cache, with_video=True)
    rows = [
        {"id": "sample-a", "caption": "first", "source": "/video/a.mp4"},
        {"id": "sample-b", "caption": "second", "source": "/video/b.mp4"},
    ]
    with (cache / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    sample = _load(cache, include_metadata=True)[1]

    assert sample["sample_id"] == "sample-b"
    assert sample["caption"] == "second"
    assert sample["source_path"] == "/video/b.mp4"


def test_mmaudio_feature_cache_discovers_nested_shards(tmp_path: Path) -> None:
    first = tmp_path / "worker_00000" / "shard_000000"
    second = tmp_path / "worker_00001" / "shard_000000"
    _write_cache(first, with_video=True)
    _write_cache(second, with_video=True)

    assert _expand_cache_root(tmp_path) == [first, second]


def test_mmaudio_latent_stats_match_official_reduction(tmp_path: Path) -> None:
    first = tmp_path / "worker_00000" / "shard_000000"
    second = tmp_path / "worker_00001" / "shard_000000"
    _write_cache(first, with_video=True)
    _write_cache(second, with_video=True)

    first_tensors = tensordict.TensorDict.load_memmap(first)["mean"]
    second_tensors = tensordict.TensorDict.load_memmap(second)["mean"]
    complete = torch.cat([first_tensors, second_tensors], dim=0)
    expected_mean = complete.mean(dim=(0, 1), keepdim=True)
    expected_std = complete.std(dim=(0, 1), keepdim=True)

    actual_mean, actual_std = compute_mmaudio_latent_stats(
        tmp_path,
        latent_seq_len=4,
        latent_dim=3,
        chunk_size=1,
    )

    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(actual_std, expected_std)
