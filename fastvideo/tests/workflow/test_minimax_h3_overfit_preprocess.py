# SPDX-License-Identifier: Apache-2.0
"""Verify that MiniMax H3 preprocessing preserves one synchronized text-to-video-and-audio sample."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2va
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit import (
    TRAINING_VIDEO_NAME,
    build_parquet_record,
    load_crush_smol_training_sample,
    validate_preprocessed_training_data,
    write_parquet,
)

_TRAINING_CAPTION = ("A watermelon wearing a helmet is crushed by a hydraulic press, "
                     "causing it to flatten and burst open.")


def test_load_crush_smol_training_sample_selects_fixed_record(tmp_path: Path) -> None:
    """Verify that preprocessing reads the selected caption from the dataset manifest."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / TRAINING_VIDEO_NAME
    video_path.touch()
    manifest_path = tmp_path / "videos2caption.json"
    manifest_path.write_text(
        json.dumps([
            {
                "path": "another-video.mp4",
                "cap": ["Another caption."],
            },
            {
                "path": TRAINING_VIDEO_NAME,
                "cap": [_TRAINING_CAPTION],
            },
        ]))

    selected_video_path, caption = load_crush_smol_training_sample(manifest_path, video_dir)

    assert selected_video_path == video_path
    assert caption == _TRAINING_CAPTION


def test_load_crush_smol_training_sample_rejects_duplicate_record(tmp_path: Path) -> None:
    """Verify that an ambiguous manifest cannot silently select a training target."""
    manifest_path = tmp_path / "videos2caption.json"
    duplicate_record = {
        "path": TRAINING_VIDEO_NAME,
        "cap": [_TRAINING_CAPTION],
    }
    manifest_path.write_text(json.dumps([duplicate_record, duplicate_record]))

    with pytest.raises(ValueError, match="exactly one"):
        load_crush_smol_training_sample(manifest_path, tmp_path / "videos")


def test_write_parquet_replaces_stale_rows_and_round_trips_joint_latents(tmp_path: Path) -> None:
    """Keep one authoritative sample and preserve every tensor through collation."""
    video_latents = torch.arange(24 * 2 * 4 * 4, dtype=torch.float32).reshape(24, 2, 4, 4)
    audio_latents = torch.arange(2 * 32 * 3, dtype=torch.float32).reshape(2, 32, 3)
    text_embedding = torch.arange(4 * 5120, dtype=torch.float32).reshape(4, 5120)
    record = build_parquet_record(
        file_name=TRAINING_VIDEO_NAME,
        caption=_TRAINING_CAPTION,
        video_latents=video_latents,
        audio_latents=audio_latents,
        text_embedding=text_embedding,
    )

    stale_path = tmp_path / "data_99999.parquet"
    stale_path.write_bytes(b"stale")
    output_path = write_parquet(record, tmp_path)
    assert list(tmp_path.glob("*.parquet")) == [output_path]

    loaded_record = pq.read_table(output_path).to_pylist()[0]
    batch = collate_rows_from_parquet_schema(
        [loaded_record],
        pyarrow_schema_t2va,
        text_padding_length=8,
    )

    torch.testing.assert_close(batch["vae_latent"][0], video_latents)
    torch.testing.assert_close(batch["audio_latent"][0], audio_latents)
    torch.testing.assert_close(batch["text_embedding"][0, :4], text_embedding)
    torch.testing.assert_close(
        batch["text_attention_mask"],
        torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32),
    )
    assert set(record) == set(pyarrow_schema_t2va.names)

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / TRAINING_VIDEO_NAME).touch()
    manifest_path = tmp_path / "videos2caption.json"
    manifest_path.write_text(json.dumps([{
        "path": TRAINING_VIDEO_NAME,
        "cap": [_TRAINING_CAPTION],
    }]))
    validate_preprocessed_training_data(tmp_path, manifest_path, video_dir)
