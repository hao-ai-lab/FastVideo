import json

import pytest
import torch

from fastvideo.eval import Video, samples_from_manifest
from fastvideo.eval.pool import VideoPool


def test_samples_from_manifest_resolves_media_paths(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    video = media / "sample.mp4"
    audio = media / "sample.wav"
    video.touch()
    audio.touch()
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps({
            "id": "sample",
            "video": "media/sample.mp4",
            "audio": "media/sample.wav",
            "text_prompt": "A bell rings.",
        }) + "\n",
        encoding="utf-8",
    )

    samples = samples_from_manifest(manifest)

    assert len(samples) == 1
    assert samples[0]["id"] == "sample"
    assert isinstance(samples[0]["video"], Video)
    assert samples[0]["video"].source == str(video)
    assert samples[0]["audio"] == str(audio)
    assert samples[0]["text_prompt"] == "A bell rings."


def test_samples_from_manifest_rejects_rows_without_generated_media(tmp_path):
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps({
            "id": "sample",
            "text_prompt": "A bell rings."
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="needs at least video or audio"):
        samples_from_manifest(manifest)


def test_samples_from_manifest_rejects_empty_manifest(tmp_path):
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is empty"):
        samples_from_manifest(manifest)


def test_video_pool_propagates_path_video_fps(monkeypatch):
    frames = torch.zeros(2, 3, 4, 4)
    monkeypatch.setattr(
        "fastvideo.eval.io.video.probe_video_fps",
        lambda _source: 29.97,
    )
    monkeypatch.setattr(
        "fastvideo.eval.io.video.load_video",
        lambda _source: frames,
    )
    video = Video(source="sample.mp4")

    decoded = VideoPool([])._decode({"video": video})

    assert decoded["fps"] == pytest.approx(29.97)
    assert decoded["video"].frames is frames
