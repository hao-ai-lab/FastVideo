# SPDX-License-Identifier: Apache-2.0

import os

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
