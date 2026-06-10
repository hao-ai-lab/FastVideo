# SPDX-License-Identifier: Apache-2.0
"""Unit tests for parquet map-style dataset path parsing."""

from __future__ import annotations

import pickle

from fastvideo.dataset import parquet_dataset_map_style as parquet_dataset
from fastvideo.dataset.parquet_dataset_map_style import _parse_data_path_specs


def test_parse_data_path_specs_accepts_old_repeat_string() -> None:
    # Dataset parsing keeps compatibility with the old "path:repeat" string
    # form used by existing training configs.
    assert _parse_data_path_specs("data/path1:2,data/path2:1") == [
        ("data/path1", 2),
        ("data/path2", 1),
    ]


def test_parse_data_path_specs_accepts_yaml_mapping() -> None:
    # New YAML mapping form should reach the dataset layer as path -> repeat
    # and parse to the same internal spec representation.
    assert _parse_data_path_specs({
        "data/path1": 1,
        "data/path2": 2,
    }) == [
        ("data/path1", 1),
        ("data/path2", 2),
    ]


def test_parse_data_path_specs_accepts_path_list() -> None:
    assert _parse_data_path_specs(["data/a", "data/b"]) == [
        ("data/a", 1),
        ("data/b", 1),
    ]


def test_get_parquet_files_and_length_repeats_single_path(tmp_path, monkeypatch) -> None:
    # get_parquet_files_and_length applies repeat counts after reading the
    # per-root parquet cache, so a repeated root duplicates both names and rows.
    dataset_root = tmp_path / "dataset"
    cache_dir = dataset_root / "map_style_cache"
    cache_dir.mkdir(parents=True)
    parquet_file = dataset_root / "sample.parquet"
    parquet_file.touch()
    with (cache_dir / "file_info.pkl").open("wb") as f:
        pickle.dump(((str(parquet_file),), (7,)), f)

    class DummyWorldGroup:

        def barrier(self) -> None:
            return None

    monkeypatch.setattr(parquet_dataset, "get_world_rank", lambda: 0)
    monkeypatch.setattr(parquet_dataset, "get_world_group", DummyWorldGroup)

    file_names, lengths = parquet_dataset.get_parquet_files_and_length({
        str(dataset_root): 2,
    })

    assert file_names == (str(parquet_file), str(parquet_file))
    assert lengths == (7, 7)
