# SPDX-License-Identifier: Apache-2.0
"""Unit tests for parquet map-style dataset path parsing."""

from __future__ import annotations

import pickle

import pytest

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


def test_parse_data_path_specs_drops_non_positive_repeats() -> None:
    assert _parse_data_path_specs({"data/a": 0, "data/b": -1, "data/c": 2}) == [
        ("data/c", 2),
    ]
    assert _parse_data_path_specs("data/a:0,data/b:2") == [("data/b", 2)]


def test_parse_data_path_specs_rejects_malformed_repeats() -> None:
    # Repeat counts must parse as ints; anything else is an explicit error
    # rather than a silently mis-weighted dataset.
    with pytest.raises(ValueError):
        _parse_data_path_specs("data/a:abc")
    with pytest.raises(ValueError):
        _parse_data_path_specs("data/a:1.5")
    with pytest.raises(ValueError):
        _parse_data_path_specs({"data/a": "abc"})  # type: ignore[dict-item]


class _DummyWorldGroup:

    def barrier(self) -> None:
        return None


def _write_root_cache(dataset_root, filename: str, length: int) -> str:
    cache_dir = dataset_root / "map_style_cache"
    cache_dir.mkdir(parents=True)
    parquet_file = dataset_root / filename
    parquet_file.touch()
    with (cache_dir / "file_info.pkl").open("wb") as f:
        pickle.dump(((str(parquet_file),), (length,)), f)
    return str(parquet_file)


def test_get_parquet_files_and_length_repeats_single_path(tmp_path, monkeypatch) -> None:
    # get_parquet_files_and_length applies repeat counts after reading the
    # per-root parquet cache, so a repeated root duplicates both names and rows.
    dataset_root = tmp_path / "dataset"
    parquet_file = _write_root_cache(dataset_root, "sample.parquet", 7)

    monkeypatch.setattr(parquet_dataset, "get_world_rank", lambda: 0)
    monkeypatch.setattr(parquet_dataset, "get_world_group", _DummyWorldGroup)

    file_names, lengths = parquet_dataset.get_parquet_files_and_length({
        str(dataset_root): 2,
    })

    assert file_names == (parquet_file, parquet_file)
    assert lengths == (7, 7)


def test_get_parquet_files_and_length_mixes_roots_and_resorts(tmp_path, monkeypatch) -> None:
    # Multiple roots are expanded per repeat count and then globally re-sorted
    # by filename, so the mix order is independent of the mapping order.
    root_a = tmp_path / "dataset_a"
    root_b = tmp_path / "dataset_b"
    file_a = _write_root_cache(root_a, "a.parquet", 5)
    file_b = _write_root_cache(root_b, "b.parquet", 9)

    monkeypatch.setattr(parquet_dataset, "get_world_rank", lambda: 0)
    monkeypatch.setattr(parquet_dataset, "get_world_group", _DummyWorldGroup)

    file_names, lengths = parquet_dataset.get_parquet_files_and_length({
        str(root_b): 1,
        str(root_a): 2,
    })

    assert file_names == (file_a, file_a, file_b)
    assert lengths == (5, 5, 9)


def test_get_parquet_files_and_length_raises_when_all_repeats_dropped() -> None:
    # Zero/negative repeats are dropped at parse time; if that leaves nothing
    # to read, the mix branch fails loudly instead of yielding an empty dataset.
    with pytest.raises(FileNotFoundError):
        parquet_dataset.get_parquet_files_and_length({"data/a": 0})
