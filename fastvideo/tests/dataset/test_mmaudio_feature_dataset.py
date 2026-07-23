# SPDX-License-Identifier: Apache-2.0
"""Tests for the MMAudio-compatible precomputed feature cache."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fastvideo.dataset.mmaudio_feature_dataset import (
    MMAudioFeatureDataset,
    _expand_cache_root,
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


def _load(path: Path) -> MMAudioFeatureDataset:
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


def test_mmaudio_feature_cache_discovers_nested_shards(tmp_path: Path) -> None:
    first = tmp_path / "worker_00000" / "shard_000000"
    second = tmp_path / "worker_00001" / "shard_000000"
    _write_cache(first, with_video=True)
    _write_cache(second, with_video=True)

    assert _expand_cache_root(tmp_path) == [first, second]
