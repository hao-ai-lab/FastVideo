"""End-to-end tests for T2V preprocessing pipelines.

Downloads a small test dataset, runs old and new preprocessing pipelines
via torchrun, and validates the output parquet files structurally.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "1"
MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OLD_ENTRY_FILE = "fastvideo/pipelines/preprocess/v1_preprocess.py"
NEW_ENTRY_FILE = ("fastvideo/pipelines/preprocess/v1_preprocessing_new.py")
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "cats"
PREPROCESSED_DIR_OLD = DATA_DIR / "cats_preprocessed_t2v"
PREPROCESSED_DIR_NEW = DATA_DIR / "cats_preprocessed_t2v_new"

EXPECTED_T2V_COLUMNS = {
    "id",
    "file_name",
    "caption",
    "media_type",
    "width",
    "height",
    "num_frames",
    "duration_sec",
    "fps",
    "vae_latent_bytes",
    "vae_latent_shape",
    "vae_latent_dtype",
    "text_embedding_bytes",
    "text_embedding_shape",
    "text_embedding_dtype",
}


def _download_data():
    """Download the small cats overfit dataset from HuggingFace."""
    os.makedirs(DATA_DIR, exist_ok=True)
    snapshot_download(
        repo_id="wlsaidhi/cats-overfit-merged",
        local_dir=str(RAW_DATA_DIR),
        repo_type="dataset",
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    assert RAW_DATA_DIR.exists(), (
        f"Download failed: {RAW_DATA_DIR} does not exist")


def _run_old_preprocessing():
    """Run the old T2V preprocessing pipeline via torchrun."""
    if PREPROCESSED_DIR_OLD.exists():
        shutil.rmtree(PREPROCESSED_DIR_OLD)

    cmd = [
        "torchrun",
        "--nnodes",
        NUM_NODES,
        "--nproc_per_node",
        NUM_GPUS_PER_NODE,
        OLD_ENTRY_FILE,
        "--model_path",
        MODEL_PATH,
        "--data_merge_path",
        os.path.join(RAW_DATA_DIR, "merge_1_sample.txt"),
        "--preprocess_video_batch_size",
        "1",
        "--max_height",
        "480",
        "--max_width",
        "832",
        "--num_frames",
        "77",
        "--dataloader_num_workers",
        "0",
        "--output_dir",
        str(PREPROCESSED_DIR_OLD),
        "--train_fps",
        "16",
        "--samples_per_file",
        "1",
        "--flush_frequency",
        "1",
        "--video_length_tolerance_range",
        "5",
        "--preprocess_task",
        "t2v",
    ]  # fmt: skip

    subprocess.run(cmd, check=True)


def _run_new_preprocessing():
    """Run the new T2V preprocessing pipeline via torchrun."""
    if PREPROCESSED_DIR_NEW.exists():
        shutil.rmtree(PREPROCESSED_DIR_NEW)

    cmd = [
        "torchrun",
        "--nnodes",
        NUM_NODES,
        "--nproc_per_node",
        NUM_GPUS_PER_NODE,
        NEW_ENTRY_FILE,
        "--model-path",
        MODEL_PATH,
        "--mode",
        "preprocess",
        "--workload-type",
        "t2v",
        "--preprocess.dataset-path",
        str(RAW_DATA_DIR),
        "--preprocess.dataset-type",
        "merged",
        "--preprocess.dataset-output-dir",
        str(PREPROCESSED_DIR_NEW),
        "--preprocess.video-loader-type",
        "torchvision",
        "--preprocess.preprocess-video-batch-size",
        "1",
        "--preprocess.max-height",
        "480",
        "--preprocess.max-width",
        "832",
        "--preprocess.num-frames",
        "77",
        "--preprocess.dataloader-num-workers",
        "0",
        "--preprocess.train-fps",
        "16",
        "--preprocess.samples-per-file",
        "1",
        "--preprocess.flush-frequency",
        "1",
        "--preprocess.video-length-tolerance-range",
        "5",
    ]  # fmt: skip

    subprocess.run(cmd, check=True)


def _read_first_parquet(parquet_dir):
    """Read and return the first parquet table from a directory."""
    assert parquet_dir.exists(), (
        f"Expected output dir not found: {parquet_dir}")
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    assert len(parquet_files) >= 1, (f"No parquet files in {parquet_dir}")
    return pq.read_table(str(parquet_files[0]))


def _validate_parquet_t2v(table, label=""):
    """Validate a parquet table has the expected T2V schema and content."""
    prefix = f"[{label}] " if label else ""

    # Schema
    actual_columns = set(table.schema.names)
    missing = EXPECTED_T2V_COLUMNS - actual_columns
    assert not missing, f"{prefix}Missing columns: {missing}"

    # Row count
    assert table.num_rows >= 1, f"{prefix}Parquet has zero rows"

    # Per-row validation
    for i in range(table.num_rows):
        row = {
            col: table.column(col)[i].as_py()
            for col in EXPECTED_T2V_COLUMNS
        }

        assert len(row["vae_latent_bytes"]) > 0, (
            f"{prefix}Row {i}: vae_latent_bytes is empty")
        assert len(row["vae_latent_shape"]) == 4, (
            f"{prefix}Row {i}: vae_latent_shape should have "
            f"4 elements (C,T,H,W), got {row['vae_latent_shape']}")
        assert len(row["text_embedding_bytes"]) > 0, (
            f"{prefix}Row {i}: text_embedding_bytes is empty")
        assert isinstance(row["caption"], str) and row["caption"], (
            f"{prefix}Row {i}: caption is empty or not a string")
        assert row["media_type"] == "video", (
            f"{prefix}Row {i}: expected media_type='video', "
            f"got '{row['media_type']}'")
        assert row["width"] > 0, (f"{prefix}Row {i}: width must be positive")
        assert row["height"] > 0, (f"{prefix}Row {i}: height must be positive")


def test_preprocessing_t2v_old():
    """Run old T2V preprocessing and validate output parquet files."""
    _download_data()
    _run_old_preprocessing()

    parquet_dir = PREPROCESSED_DIR_OLD / "combined_parquet_dataset"
    table = _read_first_parquet(parquet_dir)
    _validate_parquet_t2v(table, label="old pipeline")


def test_preprocessing_t2v_new():
    """Run new T2V preprocessing and validate output parquet files."""
    _download_data()
    _run_new_preprocessing()

    parquet_dir = PREPROCESSED_DIR_NEW / "training_dataset" / "worker_0"
    table = _read_first_parquet(parquet_dir)
    _validate_parquet_t2v(table, label="new pipeline")


def test_preprocessing_pipelines_match():
    """Compare old and new pipeline outputs structurally."""
    _download_data()
    _run_old_preprocessing()
    _run_new_preprocessing()

    old_dir = PREPROCESSED_DIR_OLD / "combined_parquet_dataset"
    new_dir = PREPROCESSED_DIR_NEW / "training_dataset" / "worker_0"
    old_table = _read_first_parquet(old_dir)
    new_table = _read_first_parquet(new_dir)

    # Both must have the expected columns
    for name, tbl in [("old", old_table), ("new", new_table)]:
        actual = set(tbl.schema.names)
        missing = EXPECTED_T2V_COLUMNS - actual
        assert not missing, f"{name} pipeline missing columns: {missing}"

    # Same number of rows
    assert old_table.num_rows == new_table.num_rows, (
        f"Row count mismatch: old={old_table.num_rows}, "
        f"new={new_table.num_rows}")

    # Per-row structural comparison (not exact bytes — fp diffs expected)
    for i in range(old_table.num_rows):
        old_row = {
            col: old_table.column(col)[i].as_py()
            for col in EXPECTED_T2V_COLUMNS
        }
        new_row = {
            col: new_table.column(col)[i].as_py()
            for col in EXPECTED_T2V_COLUMNS
        }

        assert old_row["vae_latent_shape"] == new_row["vae_latent_shape"], (
            f"Row {i}: vae_latent_shape mismatch: "
            f"old={old_row['vae_latent_shape']}, "
            f"new={new_row['vae_latent_shape']}")
        assert (
            old_row["text_embedding_shape"] == new_row["text_embedding_shape"]
        ), (f"Row {i}: text_embedding_shape mismatch: "
            f"old={old_row['text_embedding_shape']}, "
            f"new={new_row['text_embedding_shape']}")
        assert old_row["vae_latent_dtype"] == new_row["vae_latent_dtype"], (
            f"Row {i}: vae_latent_dtype mismatch")
        assert (
            old_row["text_embedding_dtype"] == new_row["text_embedding_dtype"]
        ), (f"Row {i}: text_embedding_dtype mismatch")
        assert old_row["media_type"] == new_row["media_type"], (
            f"Row {i}: media_type mismatch")


if __name__ == "__main__":
    test_preprocessing_t2v_old()
    test_preprocessing_t2v_new()
    test_preprocessing_pipelines_match()
