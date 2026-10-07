# SPDX-License-Identifier: Apache-2.0
"""Tests for shared raw V2A preprocessing infrastructure."""

from __future__ import annotations

import json
import math
import sys
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from fastvideo.configs.configs import VideoLoaderType
from fastvideo.dataset.mmaudio_feature_dataset import _expand_cache_root
from fastvideo.dataset.v2a_feature_cache import V2AFeatureShardWriter
from fastvideo.dataset.vggsound import VGGSoundDataset
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.pipelines.basic.mmaudio import stages as basic_mmaudio_stages
from fastvideo.pipelines.pipeline_registry import PipelineType, _PipelineRegistry
from fastvideo.pipelines.preprocess.mmaudio import stages as mmaudio_stages
from fastvideo.pipelines.preprocess.mmaudio import torio_media_reader
from fastvideo.pipelines.preprocess.mmaudio.torio_media_reader import (
    MMAudioTorioRowPreprocessor,
    PREPROCESSED_MEDIA_KEY,
    PREPROCESS_ERROR_KEY,
    preprocess_mmaudio_media_with_torio,
)
from fastvideo.workflow.preprocess.preprocess_workflow_v2a import (
    PreprocessWorkflowV2A, V2AForwardBatchBuilder)

pytest.importorskip("tensordict")


class _GenericT2VPreprocessPipeline:
    pass


class _FamilyPreprocessPipeline:
    pass


def _write_fractional_fps_video(
    path: Path,
    *,
    duration_s: float,
    frame_rate: Fraction = Fraction(30_000, 1_001),
) -> None:
    av = pytest.importorskip("av")
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=frame_rate)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        time_base = Fraction(frame_rate.denominator, frame_rate.numerator)
        frame_count = math.ceil(duration_s * float(frame_rate))
        for index in range(frame_count):
            pixels = np.full((32, 32, 3), index % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


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


def test_mmaudio_torio_loader_is_an_explicit_config_choice() -> None:
    assert VideoLoaderType.from_string("torio") is VideoLoaderType.TORIO


def test_mmaudio_torio_reader_preserves_reference_media_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, float | None]] = []

    class _StreamInfo:
        sample_rate = 44_100

    class _FakeStreamingMediaDecoder:
        def __init__(self, path: str) -> None:
            assert path == "sample.mp4"

        def add_basic_video_stream(
            self,
            *,
            frames_per_chunk: int,
            frame_rate: float,
            format: str,
        ) -> None:
            assert format == "rgb24"
            calls.append(("video", frames_per_chunk, frame_rate))

        def add_basic_audio_stream(self, *, frames_per_chunk: int) -> None:
            calls.append(("audio", frames_per_chunk, None))

        def fill_buffer(self) -> None:
            pass

        def pop_chunks(self):
            clip = torch.arange(8 * 3 * 16 * 20, dtype=torch.uint8).reshape(8, 3, 16, 20)
            sync = torch.arange(25 * 3 * 16 * 20, dtype=torch.uint8).reshape(25, 3, 16, 20)
            audio = torch.stack([
                torch.linspace(-0.5, 0.5, 44_100),
                torch.linspace(-0.25, 0.25, 44_100),
            ], dim=1)
            return clip, sync, audio

        def get_out_stream_info(self, index: int) -> _StreamInfo:
            assert index == 2
            return _StreamInfo()

    torio_module = types.ModuleType("torio")
    torio_module.__path__ = []
    io_module = types.ModuleType("torio.io")
    io_module.StreamingMediaDecoder = _FakeStreamingMediaDecoder
    torio_module.io = io_module
    monkeypatch.setitem(sys.modules, "torio", torio_module)
    monkeypatch.setitem(sys.modules, "torio.io", io_module)

    audio, clip, sync, duration = preprocess_mmaudio_media_with_torio(
        "sample.mp4",
        duration_s=1.0,
        target_sample_rate=44_100,
        target_samples=44_100,
        normalize_audio=True,
        clip_size=32,
        sync_size=24,
    )

    assert calls == [
        ("video", 8, 8.0),
        ("video", 25, 25.0),
        ("audio", 2**30, None),
    ]
    assert audio.shape == (44_100,)
    torch.testing.assert_close(audio.abs().max(), torch.tensor(0.95))
    assert clip.shape == (8, 3, 32, 32)
    assert sync.shape == (25, 3, 24, 24)
    assert duration == 1.0


def test_mmaudio_torio_row_preprocessor_returns_worker_ready_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    expected = (
        torch.ones(16),
        torch.ones(8, 3, 4, 4),
        torch.ones(25, 3, 4, 4),
        1.0,
    )
    monkeypatch.setattr(
        torio_media_reader,
        "preprocess_mmaudio_media_with_torio",
        lambda *_args, **_kwargs: expected,
    )
    preprocessor = MMAudioTorioRowPreprocessor(
        duration_s=1.0,
        target_sample_rate=16,
        target_samples=16,
        normalize_audio=True,
        clip_size=4,
        sync_size=4,
    )

    output = preprocessor({
        "id": "sample",
        "caption": "sound",
        "video_path": str(video_path),
    })

    assert PREPROCESS_ERROR_KEY not in output
    media = output[PREPROCESSED_MEDIA_KEY]
    assert media["audio"] is expected[0]
    assert media["clip_frames"] is expected[1]
    assert media["sync_frames"] is expected[2]
    assert media["effective_duration"] == 1.0


def test_v2a_batch_builder_preserves_preprocessed_media_alignment() -> None:
    media = {
        "audio": torch.ones(1),
        "clip_frames": torch.ones(1),
        "sync_frames": torch.ones(1),
        "effective_duration": 1.0,
    }
    rows = [{
        "id": "good",
        "caption": "good sound",
        "video_path": "good.mp4",
        PREPROCESSED_MEDIA_KEY: media,
    }, {
        "id": "bad",
        "caption": "bad sound",
        "video_path": "bad.mp4",
        PREPROCESS_ERROR_KEY: "decode failed",
    }]

    batch = V2AForwardBatchBuilder(seed=123)(rows)

    assert batch.extra[PREPROCESSED_MEDIA_KEY] == [media, None]
    assert batch.extra[PREPROCESS_ERROR_KEY] == [None, "decode failed"]


def test_v2a_torio_thread_prefetch_preserves_batch_order() -> None:
    workflow = object.__new__(PreprocessWorkflowV2A)
    workflow.training_dataloader = [
        [{"id": "0"}, {"id": "1"}],
        [{"id": "2"}, {"id": "3"}],
        [{"id": "4"}],
    ]
    workflow.torio_decode_workers = 2
    workflow.torio_row_preprocessor = lambda row: {
        **row,
        "decoded": True,
    }

    batches = list(workflow._iter_preprocessed_rows())

    assert [[row["id"] for row in batch] for batch in batches] == [
        ["0", "1"],
        ["2", "3"],
        ["4"],
    ]
    assert all(row["decoded"] for batch in batches for row in batch)


def test_mmaudio_ffmpeg_sampler_recovers_fractional_fps_boundary(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "fractional-fps.mp4"
    _write_fractional_fps_video(video_path, duration_s=10.0)

    clip_frames, sync_frames = basic_mmaudio_stages._read_frames_with_fps_filters(
        video_path,
        (8.0, 25.0),
        start_s=0.0,
        end_s=8.0,
    )

    assert clip_frames.shape == (64, 32, 32, 3)
    assert sync_frames.shape == (200, 32, 32, 3)


def test_mmaudio_ffmpeg_sampler_does_not_pad_short_video(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "short.mp4"
    _write_fractional_fps_video(video_path, duration_s=7.0)

    clip_frames, sync_frames = basic_mmaudio_stages._read_frames_with_fps_filters(
        video_path,
        (8.0, 25.0),
        start_s=0.0,
        end_s=8.0,
    )

    assert clip_frames.shape[0] < 64
    assert sync_frames.shape[0] < 200


def test_mmaudio_inference_keeps_timestamp_sampler_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    sampled_frames = [
        np.zeros((8, 8, 8, 3), dtype=np.uint8),
        np.zeros((25, 8, 8, 3), dtype=np.uint8),
    ]

    def timestamp_sampler(*_args, **_kwargs):
        calls.append("timestamp")
        return sampled_frames

    def ffmpeg_sampler(*_args, **_kwargs):
        calls.append("ffmpeg")
        return sampled_frames

    monkeypatch.setattr(
        basic_mmaudio_stages,
        "_read_frames_at_fps",
        timestamp_sampler,
    )
    monkeypatch.setattr(
        basic_mmaudio_stages,
        "_read_frames_with_fps_filters",
        ffmpeg_sampler,
    )

    basic_mmaudio_stages.preprocess_mmaudio_video(
        "unused.mp4",
        duration_s=1.0,
        clip_size=8,
        sync_size=8,
    )
    basic_mmaudio_stages.preprocess_mmaudio_video(
        "unused.mp4",
        duration_s=1.0,
        clip_size=8,
        sync_size=8,
        use_ffmpeg_fps_filter=True,
    )

    assert calls == ["timestamp", "ffmpeg"]


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
