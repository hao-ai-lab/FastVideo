# SPDX-License-Identifier: Apache-2.0
"""Tests for shared raw V2A preprocessing infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fastvideo.dataset.mmaudio_feature_dataset import _expand_cache_root
from fastvideo.dataset.v2a_feature_cache import V2AFeatureShardWriter
from fastvideo.dataset.vggsound import VGGSoundDataset
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.pipelines.pipeline_registry import PipelineType, _PipelineRegistry
from fastvideo.pipelines.preprocess.mmaudio import stages as mmaudio_stages

pytest.importorskip("tensordict")


class _GenericT2VPreprocessPipeline:
    pass


class _FamilyPreprocessPipeline:
    pass


def test_vggsound_dataset_maps_hf_metadata_to_clip_name(tmp_path: Path) -> None:
    (tmp_path / "videos").mkdir()
    (tmp_path / "vggsound.csv").write_text(
        "abc123,7,a train sound,train\nxyz789,42,a test sound,test\n",
        encoding="utf-8",
    )

    dataset = VGGSoundDataset(tmp_path, split="train")

    assert len(dataset) == 1
    assert dataset[0] == {
        "id": "abc123_000007",
        "caption": "a train sound",
        "video_path": str(tmp_path / "videos" / "abc123_000007.mp4"),
    }


def test_vggsound_dataset_reads_filtered_caption_manifest(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "video-one_000010.mp4").touch()
    manifest = tmp_path / "vgg-val-filtered-caption.tsv"
    manifest.write_text(
        "id\tlabel\n"
        "video-one_000010\tA detailed description of the audible event.\n"
        "missing_000020\tThis row has no corresponding video.\n",
        encoding="utf-8",
    )

    dataset = VGGSoundDataset(tmp_path, split="val", metadata_path=manifest)

    assert len(dataset) == 1
    assert dataset[0] == {
        "id": "video-one_000010",
        "caption": "A detailed description of the audible event.",
        "video_path": str(videos / "video-one_000010.mp4"),
    }


def test_mmaudio_audio_normalization_matches_split_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    waveform = torch.tensor([[0.25, -0.5, 0.125, 0.0]], dtype=torch.float32)
    monkeypatch.setattr(
        mmaudio_stages.torchaudio,
        "load",
        lambda _path: (waveform.clone(), 44_100),
    )

    train_audio = mmaudio_stages.MMAudioFeatureExtractionStage._load_audio(
        "sample.mp4",
        target_sample_rate=44_100,
        target_samples=4,
        normalize_audio=True,
    )
    eval_audio = mmaudio_stages.MMAudioFeatureExtractionStage._load_audio(
        "sample.mp4",
        target_sample_rate=44_100,
        target_samples=4,
        normalize_audio=False,
    )

    torch.testing.assert_close(train_audio.abs().max(), torch.tensor(0.95))
    torch.testing.assert_close(eval_audio, waveform[0])


def test_v2a_writer_creates_discoverable_resumable_shards(tmp_path: Path) -> None:
    writer = V2AFeatureShardWriter(tmp_path, rank=0, samples_per_shard=1)
    features = {
        "mean": torch.randn(1, 4, 3),
        "std": torch.rand(1, 4, 3),
        "clip_features": torch.randn(1, 6, 8),
        "sync_features": torch.randn(1, 9, 10),
        "text_features": torch.randn(1, 5, 7),
    }
    writer.append(features, [{"id": "sample-1", "caption": "sound"}])
    writer.close()

    shards = _expand_cache_root(tmp_path)
    assert shards == [tmp_path / "worker_00000" / "shard_000000"]
    metadata = json.loads((shards[0] / "samples.jsonl").read_text().strip())
    assert metadata["id"] == "sample-1"

    resumed = V2AFeatureShardWriter(tmp_path, rank=0, samples_per_shard=1)
    assert resumed.contains("sample-1")
    resumed.append(features, [{"id": "sample-1", "caption": "duplicate"}])
    resumed.close()
    assert len(_expand_cache_root(tmp_path)) == 1


def test_family_preprocessing_is_enabled_only_for_v2a() -> None:
    registry = _PipelineRegistry({
        PipelineType.PREPROCESS.value: {
            "PreprocessPipelineT2V": _GenericT2VPreprocessPipeline,
            "ExamplePreprocessPipeline": _FamilyPreprocessPipeline,
        }
    })

    assert registry._load_preprocess_pipeline_cls("ExamplePipeline", WorkloadType.V2A) is _FamilyPreprocessPipeline
    assert registry._load_preprocess_pipeline_cls("ExamplePipeline", WorkloadType.T2V) is _GenericT2VPreprocessPipeline
