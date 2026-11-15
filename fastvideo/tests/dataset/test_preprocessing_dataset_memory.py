# SPDX-License-Identifier: Apache-2.0
"""VideoCaptionMergedDataset must not keep decoded pixels on its stored metadata."""

import argparse
import json

import torch

from fastvideo.dataset import getdataset


def _dataset(tmp_path, clips: int):
    videos = tmp_path / "videos"
    videos.mkdir()
    annotations = []
    for i in range(clips):
        (videos / f"clip{i}.mp4").write_bytes(b"")
        annotations.append({"path": f"clip{i}.mp4", "cap": ["a caption"], "resolution": {"width": 32, "height": 32},
                            "fps": 8, "duration": 1.0, "num_frames": 8})
    (tmp_path / "annotation.json").write_text(json.dumps(annotations))
    (tmp_path / "merge.txt").write_text(f"{videos},{tmp_path / 'annotation.json'}\n")
    args = argparse.Namespace(model_path=str(tmp_path / "no-model"), cache_dir=str(tmp_path), num_frames=8,
                              train_fps=8, speed_factor=1.0, video_length_tolerance_range=5, drop_short_ratio=1.0,
                              text_max_length=32, training_cfg_rate=0.0, max_height=32, max_width=32,
                              do_temporal_sample=False, data_merge_path=str(tmp_path / "merge.txt"), seed=0)
    return getdataset(args)


def test_iterating_does_not_retain_pixels_on_processed_batches(tmp_path, monkeypatch) -> None:
    dataset = _dataset(tmp_path, clips=3)

    def fake_decode(batch):
        batch.pixel_values = torch.zeros(3, 8, 32, 32)
        return batch

    monkeypatch.setattr(dataset.video_transform_stage, "process", fake_decode)

    items = list(dataset)

    assert len(items) == 3 and all(item["pixel_values"] is not None for item in items)
    assert all(batch.pixel_values is None for batch in dataset.processed_batches)
