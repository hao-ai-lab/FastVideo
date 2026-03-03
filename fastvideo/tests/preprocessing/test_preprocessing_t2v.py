"""End-to-end test for T2V preprocessing pipeline.

Downloads a small test dataset, runs the preprocessing pipeline via
torchrun, and validates the output parquet files structurally.
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
PREPROCESSING_ENTRY_FILE = ("fastvideo/pipelines/preprocess/v1_preprocess.py")
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "cats"
PREPROCESSED_DATA_DIR = DATA_DIR / "cats_preprocessed_t2v"

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


def _run_preprocessing():
    """Run the T2V preprocessing pipeline via torchrun."""
    if PREPROCESSED_DATA_DIR.exists():
        shutil.rmtree(PREPROCESSED_DATA_DIR)

    cmd = [
        "torchrun",
        "--nnodes",
        NUM_NODES,
        "--nproc_per_node",
        NUM_GPUS_PER_NODE,
        PREPROCESSING_ENTRY_FILE,
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
        str(PREPROCESSED_DATA_DIR),
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


def test_preprocessing_t2v():
    """Run T2V preprocessing and validate output parquet files."""
    _download_data()
    _run_preprocessing()

    parquet_dir = PREPROCESSED_DATA_DIR / "combined_parquet_dataset"
    assert parquet_dir.exists(), (
        f"Expected output dir not found: {parquet_dir}")

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    assert len(parquet_files) >= 1, "No parquet files produced"

    table = pq.read_table(str(parquet_files[0]))

    # -- schema validation --
    actual_columns = set(table.schema.names)
    missing = EXPECTED_T2V_COLUMNS - actual_columns
    assert not missing, f"Missing columns in parquet: {missing}"

    # -- row count --
    assert table.num_rows >= 1, "Parquet file has zero rows"

    # -- per-row content validation --
    for i in range(table.num_rows):
        row = {
            col: table.column(col)[i].as_py()
            for col in EXPECTED_T2V_COLUMNS
        }

        # VAE latent
        assert len(row["vae_latent_bytes"]) > 0, (
            f"Row {i}: vae_latent_bytes is empty")
        assert len(row["vae_latent_shape"]) == 4, (
            f"Row {i}: vae_latent_shape should have 4 elements "
            f"(C,T,H,W), got {row['vae_latent_shape']}")

        # Text embedding
        assert len(row["text_embedding_bytes"]) > 0, (
            f"Row {i}: text_embedding_bytes is empty")

        # Caption
        assert isinstance(row["caption"], str) and row["caption"], (
            f"Row {i}: caption is empty or not a string")

        # Media type
        assert row["media_type"] == "video", (
            f"Row {i}: expected media_type='video', "
            f"got '{row['media_type']}'")

        # Dimensions
        assert row["width"] > 0, (
            f"Row {i}: width must be positive, got {row['width']}")
        assert row["height"] > 0, (
            f"Row {i}: height must be positive, got {row['height']}")


if __name__ == "__main__":
    test_preprocessing_t2v()
