# SPDX-License-Identifier: Apache-2.0

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fastvideo.dataset.parquet_dataset_map_style as map_style


def test_normalize_dataset_paths_accepts_single_path() -> None:
    assert map_style._normalize_dataset_paths("/data/root") == ("/data/root",)


def test_normalize_dataset_paths_splits_pathsep_string() -> None:
    path = os.pathsep.join(["/data/latents_a", "/data/latents_b"])

    assert map_style._normalize_dataset_paths(path) == (
        "/data/latents_a",
        "/data/latents_b",
    )


def test_normalize_dataset_paths_accepts_sequence() -> None:
    assert map_style._normalize_dataset_paths([
        "/data/latents_a",
        "/data/latents_b",
    ]) == (
        "/data/latents_a",
        "/data/latents_b",
    )


def test_normalize_data_split_accepts_validation_alias() -> None:
    assert map_style._normalize_data_split("val") == "validation"


def test_normalize_data_split_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="data_split"):
        map_style._normalize_data_split("dev")


def test_validate_split_ratio_requires_positive_ratio_for_split() -> None:
    with pytest.raises(ValueError, match="validation_split_ratio"):
        map_style._validate_split_ratio("train", 0.0)


def test_validation_score_is_stable() -> None:
    assert map_style._validation_score("clip-a", 42) == map_style._validation_score(
        "clip-a", 42)
    assert map_style._validation_score("clip-a", 42) != map_style._validation_score(
        "clip-a", 43)


def test_select_split_row_indices_uses_stable_row_keys(tmp_path) -> None:
    ids = [f"clip-{idx}" for idx in range(12)]
    parquet_file = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({
            "id": ids,
            "caption": [f"caption {idx}" for idx in range(len(ids))],
        }),
        parquet_file,
    )

    validation_indices = map_style._select_split_row_indices(
        [parquet_file.as_posix()],
        [len(ids)],
        data_split="validation",
        validation_split_ratio=0.5,
        seed=42,
    )
    train_indices = map_style._select_split_row_indices(
        [parquet_file.as_posix()],
        [len(ids)],
        data_split="train",
        validation_split_ratio=0.5,
        seed=42,
    )

    expected_validation = tuple(
        idx for idx, sample_id in enumerate(ids)
        if map_style._is_validation_sample(sample_id, 0.5, 42))
    expected_train = tuple(idx for idx in range(len(ids))
                           if idx not in expected_validation)

    assert validation_indices == expected_validation
    assert train_indices == expected_train
    assert set(validation_indices).isdisjoint(train_indices)


def test_get_parquet_files_and_length_concatenates_roots_in_order(
        monkeypatch) -> None:
    calls: list[str] = []

    def fake_single_root(path: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
        calls.append(path)
        return ((f"{path}/data.parquet",), (len(calls),))

    monkeypatch.setattr(
        map_style,
        "_get_single_parquet_files_and_length",
        fake_single_root,
    )

    files, lengths = map_style.get_parquet_files_and_length([
        "/data/latents_wan_v1",
        "/data/latents_pcbas_wan_v1",
    ])

    assert calls == [
        "/data/latents_wan_v1",
        "/data/latents_pcbas_wan_v1",
    ]
    assert files == (
        "/data/latents_wan_v1/data.parquet",
        "/data/latents_pcbas_wan_v1/data.parquet",
    )
    assert lengths == (1, 2)


def test_get_parquet_split_indices_offsets_multi_root_indices(
        monkeypatch) -> None:
    single_root_calls: list[str] = []
    split_calls: list[str] = []

    def fake_single_root(path: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
        single_root_calls.append(path)
        if path.endswith("latents_a"):
            return ((f"{path}/a.parquet",), (2,))
        return ((f"{path}/b.parquet",), (3,))

    def fake_single_split(path: str,
                          file_names,
                          lengths,
                          data_split: str,
                          validation_split_ratio: float,
                          seed: int) -> tuple[int, ...]:
        split_calls.append(path)
        assert data_split == "validation"
        assert validation_split_ratio == 0.2
        assert seed == 7
        if path.endswith("latents_a"):
            return (1,)
        return (0, 2)

    monkeypatch.setattr(
        map_style,
        "_get_single_parquet_files_and_length",
        fake_single_root,
    )
    monkeypatch.setattr(
        map_style,
        "_get_single_split_row_indices",
        fake_single_split,
    )

    indices = map_style.get_parquet_split_indices(
        ["/data/latents_a", "/data/latents_b"],
        data_split="validation",
        validation_split_ratio=0.2,
        seed=7,
    )

    assert single_root_calls == ["/data/latents_a", "/data/latents_b"]
    assert split_calls == ["/data/latents_a", "/data/latents_b"]
    assert indices == (1, 2, 4)


def test_dataset_getitems_maps_split_indices(monkeypatch) -> None:

    def fake_read_row(parquet_files, global_row_idx, lengths):
        return {"id": f"row-{global_row_idx}"}

    def fake_collate(rows, parquet_schema, text_padding_length, cfg_rate, seed):
        return {
            "ids": [row["id"] for row in rows],
            "sample_indices": [row["_sample_index"] for row in rows],
        }

    monkeypatch.setattr(map_style, "read_row_from_parquet_file", fake_read_row)
    monkeypatch.setattr(map_style, "collate_rows_from_parquet_schema",
                        fake_collate)

    dataset = map_style.LatentsParquetMapStyleDataset.__new__(
        map_style.LatentsParquetMapStyleDataset)
    dataset.sample_indices = (2, 5, 8)
    dataset.parquet_files = ("data.parquet",)
    dataset.lengths = (10,)
    dataset.parquet_schema = pa.schema([pa.field("id", pa.string())])
    dataset.text_padding_length = 512
    dataset.cfg_rate = 0.0
    dataset.seed = 42

    batch = dataset.__getitems__([0, 2])

    assert batch == {
        "ids": ["row-2", "row-8"],
        "sample_indices": [2, 8],
    }
