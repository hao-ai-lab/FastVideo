# SPDX-License-Identifier: Apache-2.0
"""Immutable file-info cache admission; generated files and fake rank communication only."""
import builtins
import os
import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fastvideo.dataset import parquet_dataset_map_style as data


class FakeWorld:
    def __init__(self):
        self.rank = 0
        self.messages = []
        self.cursor = 0
        self.barriers = 0

    def broadcast_object(self, message, src):
        assert src == 0
        if self.rank == 0:
            self.messages.append(message)
            return message
        message = self.messages[self.cursor]
        self.cursor += 1
        return message

    def barrier(self):
        self.barriers += 1


@pytest.fixture
def world(monkeypatch):
    group = FakeWorld()
    monkeypatch.setattr(data, "get_world_rank", lambda: group.rank)
    monkeypatch.setattr(data, "get_world_group", lambda: group)
    monkeypatch.setattr(data, "get_world_size", lambda: 4)
    monkeypatch.setattr(data, "get_sp_world_size", lambda: 2)
    return group


def cache_fixture(root):
    root.mkdir()
    a, b = root / "a.parquet", root / "b.parquet"
    a.touch()
    b.touch()
    cache = root / "map_style_cache/file_info.pkl"
    cache.parent.mkdir()
    index = ([str(b), str(a), str(b)], [2, 1, 2])
    cache.write_bytes(pickle.dumps(index))
    return cache, index


def forbid_fallback(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Required cache attempted fallback walking, inventory, mkdir or writing")
    original_open = builtins.open
    def read_only(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            forbidden()
        return original_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(data.os, "walk", forbidden)
    monkeypatch.setattr(data.os, "makedirs", forbidden)
    monkeypatch.setattr(data.pq, "ParquetFile", forbidden)
    monkeypatch.setattr(data.pickle, "dump", forbidden)
    monkeypatch.setattr(data, "open", read_only, raising=False)


@pytest.mark.parametrize("relative", [False, True])
def test_required_cache_preserves_bytes_order_duplicates_and_rank_result(tmp_path, monkeypatch, world, relative):
    root = tmp_path / "dataset"
    cache, index = cache_fixture(root)
    if relative:
        monkeypatch.chdir(tmp_path)
        index = ([os.path.relpath(name, tmp_path) for name in index[0]], index[1])
        cache.write_bytes(pickle.dumps(index))
    # Strict mode consumes exactly the publication, not an implicit census.
    (root / "not-in-the-index.parquet").touch()
    before = cache.read_bytes()
    with monkeypatch.context() as guard:
        forbid_fallback(guard)
        assert data.get_parquet_files_and_length(str(root), require_existing_parquet_cache=True) == index
        # Peers consume the validated snapshot, not a second cache read.
        guard.setattr(data, "_read_required_parquet_cache", lambda *a: pytest.fail("nonzero rank read cache"))
        for rank in (1, 2, 3):
            world.rank, world.cursor = rank, 0
            assert data.get_parquet_files_and_length(str(root), require_existing_parquet_cache=True) == index
    assert world.barriers == 0
    assert cache.read_bytes() == before


@pytest.mark.parametrize("defect", [
    "missing", "missing_directory", "unreadable", "pickle", "trailing", "structure", "sequences", "empty", "mismatched", "bad_path",
    "missing_file", "directory", "outside", "outside_symlink", "outside_cache", "cycle", "bool_count",
    "negative_count", "float_count", "no_rows", "inconsistent_repeat",
])
def test_required_cache_failure_reaches_all_ranks_without_fallback(tmp_path, monkeypatch, world, defect):
    root = tmp_path / "dataset"
    cache, index = cache_fixture(root)
    outside = tmp_path / "outside.parquet"
    outside.touch()
    value = index
    if defect == "missing": cache.unlink()
    elif defect == "missing_directory":
        cache.unlink()
        cache.parent.rmdir()
    elif defect == "pickle": cache.write_bytes(b"not a pickle")
    elif defect == "trailing": cache.write_bytes(cache.read_bytes() + b"unexpected data")
    elif defect == "structure": value = (index[0],)
    elif defect == "sequences": value = ("abc", [1, 2, 3])
    elif defect == "empty": value = ([], [])
    elif defect == "mismatched": value = (index[0], [1])
    elif defect == "bad_path": value = ([None], [1])
    elif defect == "missing_file": value = ([str(root / "missing.parquet")], [1])
    elif defect == "directory": value = ([str(root)], [1])
    elif defect == "outside": value = ([str(outside)], [1])
    elif defect in {"outside_symlink", "cycle"}:
        link = root / "link.parquet"
        link.symlink_to(outside if defect == "outside_symlink" else link)
        value = ([str(link)], [1])
    elif defect == "outside_cache":
        external = tmp_path / "external-cache.pkl"
        external.write_bytes(cache.read_bytes())
        cache.unlink()
        cache.symlink_to(external)
    elif defect == "bool_count": value = ([index[0][0]], [True])
    elif defect == "negative_count": value = ([index[0][0]], [-1])
    elif defect == "float_count": value = ([index[0][0]], [1.5])
    elif defect == "no_rows": value = ([index[0][0]], [0])
    elif defect == "inconsistent_repeat": value = (index[0], [2, 1, 3])
    if value is not index:
        cache.write_bytes(pickle.dumps(value))
    before = cache.read_bytes() if cache.exists() else None
    with monkeypatch.context() as guard:
        forbid_fallback(guard)
        if defect == "unreadable":
            def unreadable(*args, **kwargs):
                raise PermissionError("synthetic cache permission failure")
            guard.setattr(data, "open", unreadable, raising=False)
        errors = []
        for rank in range(4):
            world.rank, world.cursor = rank, 0
            with pytest.raises(RuntimeError, match="Required parquet cache failed validation") as error:
                data.get_parquet_files_and_length(str(root), require_existing_parquet_cache=True)
            errors.append(str(error.value))
        assert len(set(errors)) == 1
    assert world.barriers == 0
    assert (cache.read_bytes() if cache.exists() else None) == before


@pytest.mark.parametrize("missing_second", [False, True])
def test_required_cache_multi_root_recursion_preserves_legacy_merge(tmp_path, monkeypatch, world, missing_second):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    cache_a, index_a = cache_fixture(root_a)
    cache_b, index_b = cache_fixture(root_b)
    specs = {str(root_b): 1, str(root_a): 2}
    if missing_second:
        cache_a.unlink()  # second root in insertion order
    else:
        expected = data.get_parquet_files_and_length(specs)
        world.barriers = 0
    with monkeypatch.context() as guard:
        forbid_fallback(guard)
        for rank in range(4):
            world.rank, world.cursor = rank, 0
            if missing_second:
                with pytest.raises(RuntimeError, match="Required parquet cache"):
                    data.get_parquet_files_and_length(specs, require_existing_parquet_cache=True)
            else:
                assert data.get_parquet_files_and_length(specs, require_existing_parquet_cache=True) == expected
    assert world.barriers == 0


@pytest.mark.parametrize("invalid_cache", [False, True])
def test_default_cache_mode_still_scans_and_writes(tmp_path, world, invalid_cache):
    root = tmp_path / "dataset"
    root.mkdir()
    file = root / "one.parquet"
    pq.write_table(pa.table({"id": [1, 2]}), file)
    cache = root / "map_style_cache/file_info.pkl"
    if invalid_cache:
        cache.parent.mkdir()
        cache.write_bytes(b"broken")
    assert data.get_parquet_files_and_length(str(root)) == ((str(file),), (2,))
    assert pickle.loads(cache.read_bytes()) == ((str(file),), (2,))
    assert world.barriers == 1 and not world.messages




@pytest.mark.parametrize("bad", [None, 0, 1, "true", "false", 1.0, [], {}])
def test_required_cache_boolean_is_strict_in_loader(tmp_path, bad):
    with pytest.raises(ValueError, match="require_existing_parquet_cache"):
        data.get_parquet_files_and_length(str(tmp_path), require_existing_parquet_cache=bad)
