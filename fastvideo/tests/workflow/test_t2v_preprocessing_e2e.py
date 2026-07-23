import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v

CAPTION = "a deterministic preprocessing smoke test"
MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

pytestmark = [
    pytest.mark.skipif(os.environ.get("FASTVIDEO_PREPROCESSING_E2E") != "1",
                       reason="set FASTVIDEO_PREPROCESSING_E2E=1 to run the GPU integration test"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def test_t2v_preprocessing_writes_valid_parquet(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    video_dir = raw_data_dir / "videos"
    video_dir.mkdir(parents=True)

    video_name = "sample.mp4"
    source_video = Path(__file__).parents[1] / "nightly" / "reference_video_1_sample_v0.mp4"
    shutil.copy2(source_video, video_dir / video_name)
    (raw_data_dir / "videos2caption.json").write_text(
        json.dumps([{
            "path": video_name,
            "cap": CAPTION,
            "fps": 16.0,
            "num_frames": 29,
            "resolution": {
                "height": 480,
                "width": 832,
            },
        }]),
        encoding="utf-8",
    )

    output_dir = tmp_path / "preprocessed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "-m",
            "fastvideo.pipelines.preprocess.v1_preprocessing_new",
            "--model-path",
            MODEL_PATH,
            "--mode",
            "preprocess",
            "--workload-type",
            "t2v",
            "--preprocess.video-loader-type",
            "torchvision",
            "--preprocess.dataset-type",
            "merged",
            "--preprocess.dataset-path",
            str(raw_data_dir),
            "--preprocess.dataset-output-dir",
            str(output_dir),
            "--preprocess.preprocess-video-batch-size",
            "1",
            "--preprocess.dataloader-num-workers",
            "0",
            "--preprocess.max-height",
            "64",
            "--preprocess.max-width",
            "64",
            "--preprocess.num-frames",
            "17",
            "--preprocess.train-fps",
            "16",
            "--preprocess.samples-per-file",
            "1",
            "--preprocess.flush-frequency",
            "1",
            "--preprocess.video-length-tolerance-range",
            "5",
        ],
        check=True,
    )

    rank_output_dir = output_dir / "combined_parquet_dataset" / "worker_0"
    parquet_file = rank_output_dir / "worker_0" / "data_chunk_0.parquet"
    assert parquet_file.is_file()
    assert not (rank_output_dir / "worker_0" / "data_chunk_0.parquet.tmp").exists()

    table = pq.read_table(parquet_file)
    assert table.schema.equals(pyarrow_schema_t2v, check_metadata=False)
    assert table.num_rows == 1

    row = table.to_pylist()[0]
    assert row["id"] == video_name
    assert row["file_name"] == video_name
    assert row["caption"] == CAPTION
    assert row["media_type"] == "video"
    assert row["width"] == 64
    assert row["height"] == 64
    assert row["fps"] == 16.0
    assert row["duration_sec"] == pytest.approx(17 / 16)
    assert row["vae_latent_bytes"]
    assert len(row["vae_latent_shape"]) == 4
    assert all(size > 0 for size in row["vae_latent_shape"])
    assert row["num_frames"] == row["vae_latent_shape"][1]
    assert row["vae_latent_dtype"]
    assert row["text_embedding_bytes"]
    assert len(row["text_embedding_shape"]) == 2
    assert all(size > 0 for size in row["text_embedding_shape"])
    assert row["text_embedding_dtype"]
