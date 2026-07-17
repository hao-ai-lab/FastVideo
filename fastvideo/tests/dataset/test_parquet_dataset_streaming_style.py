# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fastvideo.dataset.parquet_dataset_streaming_style as streaming
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v


def _row(index: int) -> dict:
    latent = np.full((1, 1, 1, 1), index, dtype=np.float32)
    embedding = np.full((2, 3), index, dtype=np.float32)
    return {
        "id": f"row-{index}",
        "vae_latent_bytes": latent.tobytes(),
        "vae_latent_shape": list(latent.shape),
        "vae_latent_dtype": "float32",
        "text_embedding_bytes": embedding.tobytes(),
        "text_embedding_shape": list(embedding.shape),
        "text_embedding_dtype": "float32",
        "file_name": f"{index}.mp4",
        "caption": f"caption {index}",
        "media_type": "video",
        "width": 832,
        "height": 480,
        "num_frames": 121,
        "duration_sec": 5.0,
        "fps": 24.0,
        "unused_large_column": b"x" * 1024,
    }


@pytest.fixture()
def parquet_root(tmp_path: Path) -> Path:
    root = tmp_path / "shared-read-only-dataset"
    root.mkdir()
    schema = pyarrow_schema_t2v.append(pa.field("unused_large_column", pa.binary()))
    pq.write_table(pa.Table.from_pylist([_row(i) for i in range(12)], schema=schema),
                   root / "part-0.parquet", row_group_size=3)
    pq.write_table(pa.Table.from_pylist([_row(i) for i in range(12, 24)], schema=schema),
                   root / "part-1.parquet", row_group_size=3)
    return root


def _patch_dist(monkeypatch: pytest.MonkeyPatch, rank: int = 0,
                world_size: int = 1, sp_size: int = 1) -> None:
    monkeypatch.setattr(streaming.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(streaming, "get_world_rank", lambda: rank)
    monkeypatch.setattr(streaming, "get_world_size", lambda: world_size)
    monkeypatch.setattr(streaming, "get_sp_world_size", lambda: sp_size)
    monkeypatch.setattr(streaming, "_barrier", lambda: None)


def _dataset(parquet_root: Path, cache_root: Path, **kwargs):
    return streaming.LatentsParquetStreamingDataset(
        path=str(parquet_root),
        batch_size=2,
        parquet_schema=pyarrow_schema_t2v,
        manifest_path=str(cache_root / "openvid-manifest.json"),
        num_workers=0,
        text_padding_length=4,
        read_batch_size=2,
        shuffle_row_groups=False,
        **kwargs,
    )


def test_workers_zero_projects_columns_and_writes_json_manifest(
        parquet_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dist(monkeypatch)
    recorded_columns = []
    original = pq.ParquetFile.iter_batches

    def recording_iter_batches(self, *args, **kwargs):
        recorded_columns.append(tuple(kwargs["columns"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", recording_iter_batches)
    dataset = _dataset(parquet_root, tmp_path / "owned-cache")
    batch = next(iter(dataset))

    assert batch["vae_latent"].shape[0] == 2
    assert batch["info_list"][0]["id"] == "row-0"
    assert recorded_columns == [tuple(pyarrow_schema_t2v.names)]
    manifest = json.loads((tmp_path / "owned-cache" / "openvid-manifest.json").read_text())
    assert manifest["total_rows"] == 24
    assert manifest["columns"] == pyarrow_schema_t2v.names
    assert not list((tmp_path / "owned-cache").glob("*.pkl"))


def test_state_dict_resumes_at_exact_next_batch(
        parquet_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dist(monkeypatch)
    first_dataset = _dataset(parquet_root, tmp_path / "owned-cache")
    iterator = iter(first_dataset)
    assert next(iterator)["info_list"][0]["id"] == "row-0"
    state = first_dataset.state_dict()
    expected = next(iterator)["info_list"]

    resumed_dataset = _dataset(parquet_root, tmp_path / "owned-cache")
    resumed_dataset.load_state_dict(state)
    actual = next(iter(resumed_dataset))["info_list"]
    assert [item["id"] for item in actual] == [item["id"] for item in expected]


def test_odd_row_group_tails_are_carried_into_next_group(
        parquet_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dist(monkeypatch)
    dataset = _dataset(parquet_root, tmp_path / "owned-cache")
    ids = [item["id"] for batch in dataset for item in batch["info_list"]]
    assert ids == [f"row-{index}" for index in range(24)]


def test_dp_sp_shards_are_identical_within_sp_and_disjoint_across_dp(
        parquet_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids_by_rank = {}
    for rank in range(4):
        _patch_dist(monkeypatch, rank=rank, world_size=4, sp_size=2)
        dataset = _dataset(parquet_root, tmp_path / "owned-cache")
        ids_by_rank[rank] = {
            item["id"] for batch in dataset for item in batch["info_list"]
        }
    assert ids_by_rank[0] == ids_by_rank[1]
    assert ids_by_rank[2] == ids_by_rank[3]
    assert ids_by_rank[0].isdisjoint(ids_by_rank[2])
    assert len(ids_by_rank[0] | ids_by_rank[2]) == 24


def test_equal_number_of_row_groups_and_dp_shards_stays_nonempty(
        parquet_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids_by_rank = []
    for rank in range(4):
        _patch_dist(monkeypatch, rank=rank, world_size=4, sp_size=1)
        dataset = _dataset(parquet_root, tmp_path / "owned-cache")
        ids_by_rank.append({
            item["id"] for batch in dataset for item in batch["info_list"]
        })
    assert all(ids for ids in ids_by_rank)
    assert len(set().union(*ids_by_rank)) == 24


def test_manifest_must_not_be_written_inside_dataset(
        parquet_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dist(monkeypatch)
    with pytest.raises(ValueError, match="outside the read-only dataset"):
        _dataset(parquet_root, parquet_root / "cache")


def test_stateful_dataloader_restores_dataset_cursor(
        parquet_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, loader = streaming.build_parquet_streaming_style_dataloader(
        path=str(parquet_root),
        batch_size=2,
        num_data_workers=0,
        parquet_schema=pyarrow_schema_t2v,
        manifest_path=str(tmp_path / "owned-cache" / "manifest.json"),
        text_padding_length=4,
        read_batch_size=2,
        shuffle_row_groups=False,
    )
    iterator = iter(loader)
    assert next(iterator)["info_list"][0]["id"] == "row-0"
    state = loader.state_dict()
    expected = next(iterator)["info_list"]

    _, resumed = streaming.build_parquet_streaming_style_dataloader(
        path=str(parquet_root),
        batch_size=2,
        num_data_workers=0,
        parquet_schema=pyarrow_schema_t2v,
        manifest_path=str(tmp_path / "owned-cache" / "manifest.json"),
        text_padding_length=4,
        read_batch_size=2,
        shuffle_row_groups=False,
    )
    resumed.load_state_dict(state)
    actual = next(iter(resumed))["info_list"]
    assert [item["id"] for item in actual] == [item["id"] for item in expected]
