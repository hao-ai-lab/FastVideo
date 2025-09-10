from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fastvideo.dataset.dataloader.parquet_io import (
    ParquetDatasetWriter,
    records_to_table,
)
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_i2v,
    pyarrow_schema_t2v,
    pyarrow_schema_ode_trajectory_text_only,
)


def _bytes_shape_dtype(arr: np.ndarray):
    return arr.tobytes(), list(arr.shape), str(arr.dtype)


def test_records_to_table_t2v_schema(tmp_path: Path):
    schema = pyarrow_schema_t2v
    # Build one valid record
    vae = np.zeros((4, 2, 8, 8), dtype=np.float32)
    txt = np.ones((12, 16), dtype=np.float32)
    vae_b, vae_s, vae_d = _bytes_shape_dtype(vae)
    txt_b, txt_s, txt_d = _bytes_shape_dtype(txt)
    rec = {
        "id": "vid_001",
        "vae_latent_bytes": vae_b,
        "vae_latent_shape": vae_s,
        "vae_latent_dtype": vae_d,
        "text_embedding_bytes": txt_b,
        "text_embedding_shape": txt_s,
        "text_embedding_dtype": txt_d,
        "file_name": "vid_001.mp4",
        "caption": "hello",
        "media_type": "video",
        "width": 640,
        "height": 360,
        "num_frames": 2,
        "duration_sec": 0.5,
        "fps": 4.0,
    }
    table = records_to_table([rec], schema)
    assert table.schema == schema
    assert table.num_rows == 1

    out_dir = tmp_path / "t2v_out"
    writer = ParquetDatasetWriter(str(out_dir), samples_per_file=1)
    writer.append_table(table)
    written = writer.flush(num_workers=1, write_remainder=True)
    assert written == 1
    files = list(out_dir.rglob("*.parquet"))
    assert len(files) == 1
    read = pq.read_table(str(files[0]))
    assert read.schema == schema
    assert read.num_rows == 1


def test_records_to_table_i2v_schema(tmp_path: Path):
    schema = pyarrow_schema_i2v
    vae = np.zeros((4, 1, 8, 8), dtype=np.float32)
    txt = np.ones((6, 32), dtype=np.float32)
    clip = np.ones((6, 32), dtype=np.float32)
    first = np.zeros((4, 1, 8, 8), dtype=np.float32)
    pil = np.zeros((8, 8, 3), dtype=np.uint8)
    vae_b, vae_s, vae_d = _bytes_shape_dtype(vae)
    txt_b, txt_s, txt_d = _bytes_shape_dtype(txt)
    clip_b, clip_s, clip_d = _bytes_shape_dtype(clip)
    first_b, first_s, first_d = _bytes_shape_dtype(first)
    pil_b, pil_s, pil_d = _bytes_shape_dtype(pil)
    rec = {
        "id": "img_001",
        "vae_latent_bytes": vae_b,
        "vae_latent_shape": vae_s,
        "vae_latent_dtype": vae_d,
        "text_embedding_bytes": txt_b,
        "text_embedding_shape": txt_s,
        "text_embedding_dtype": txt_d,
        "clip_feature_bytes": clip_b,
        "clip_feature_shape": clip_s,
        "clip_feature_dtype": clip_d,
        "first_frame_latent_bytes": first_b,
        "first_frame_latent_shape": first_s,
        "first_frame_latent_dtype": first_d,
        "pil_image_bytes": pil_b,
        "pil_image_shape": pil_s,
        "pil_image_dtype": pil_d,
        "file_name": "img_001.png",
        "caption": "a cat",
        "media_type": "image",
        "width": 512,
        "height": 512,
        "num_frames": 1,
        "duration_sec": 0.0,
        "fps": 0.0,
    }
    table = records_to_table([rec], schema)
    assert table.schema == schema
    assert table.num_rows == 1

    out_dir = tmp_path / "i2v_out"
    writer = ParquetDatasetWriter(str(out_dir), samples_per_file=1)
    writer.append_table(table)
    written = writer.flush(num_workers=1, write_remainder=True)
    assert written == 1
    files = list(out_dir.rglob("*.parquet"))
    assert len(files) == 1
    read = pq.read_table(str(files[0]))
    assert read.schema == schema
    assert read.num_rows == 1


def test_records_to_table_ode_text_only_schema(tmp_path: Path):
    schema = pyarrow_schema_ode_trajectory_text_only
    txt = np.ones((6, 32), dtype=np.float32)
    traj = np.ones((5, 4, 2, 2), dtype=np.float32)
    tsteps = np.arange(5, dtype=np.float32)
    txt_b, txt_s, txt_d = _bytes_shape_dtype(txt)
    traj_b, traj_s, traj_d = _bytes_shape_dtype(traj)
    t_b, t_s, t_d = _bytes_shape_dtype(tsteps)
    rec = {
        "id": "text_001",
        "text_embedding_bytes": txt_b,
        "text_embedding_shape": txt_s,
        "text_embedding_dtype": txt_d,
        "trajectory_latents_bytes": traj_b,
        "trajectory_latents_shape": traj_s,
        "trajectory_latents_dtype": traj_d,
        "trajectory_timesteps_bytes": t_b,
        "trajectory_timesteps_shape": t_s,
        "trajectory_timesteps_dtype": t_d,
        "file_name": "text_001.txt",
        "caption": "prompt",
        "media_type": "text",
    }
    table = records_to_table([rec], schema)
    assert table.schema == schema
    assert table.num_rows == 1

    out_dir = tmp_path / "ode_out"
    writer = ParquetDatasetWriter(str(out_dir), samples_per_file=1)
    writer.append_table(table)
    written = writer.flush(num_workers=1, write_remainder=True)
    assert written == 1
    files = list(out_dir.rglob("*.parquet"))
    assert len(files) == 1
    read = pq.read_table(str(files[0]))
    assert read.schema == schema
    assert read.num_rows == 1


