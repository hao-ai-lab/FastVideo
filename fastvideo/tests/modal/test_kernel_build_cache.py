# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import importlib.util
import json
from pathlib import Path

import pytest


def _load_kernel_build_cache():
    module_path = Path(__file__).with_name("kernel_build_cache.py")
    spec = importlib.util.spec_from_file_location("kernel_build_cache_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kernel_build_cache = _load_kernel_build_cache()


def _patch_stable_metadata(monkeypatch) -> None:
    monkeypatch.delenv("GPU_BACKEND", raising=False)
    monkeypatch.delenv("CMAKE_ARGS", raising=False)
    monkeypatch.setattr(kernel_build_cache, "_kernel_source_hash", lambda repo_root: {"source_hash": "source"})
    monkeypatch.setattr(kernel_build_cache, "_read_kernel_version", lambda repo_root: "0.3.2")
    monkeypatch.setattr(
        kernel_build_cache,
        "_torch_metadata",
        lambda: {
            "torch_version": "2.9.0",
            "torch_cuda_version": "12.8",
            "torch_file": "/opt/venv/lib/python3.10/site-packages/torch/__init__.py",
        },
    )
    monkeypatch.setattr(
        kernel_build_cache,
        "_compiler_libc_metadata",
        lambda: {
            "compiler": {"gcc_version": "gcc 11.4.0"},
            "libc": {"ldd_version": "ldd 2.35"},
        },
    )
    monkeypatch.setattr(kernel_build_cache, "_run_optional", lambda args, cwd=None: "nvcc 12.8")


def test_cache_key_uses_resolved_arch_not_raw_env(monkeypatch, tmp_path) -> None:
    _patch_stable_metadata(monkeypatch)
    monkeypatch.setattr(kernel_build_cache, "_detect_arch_from_torch", lambda: "9.0a")

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0a")
    explicit_hopper = kernel_build_cache._build_metadata(tmp_path)

    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    detected_hopper = kernel_build_cache._build_metadata(tmp_path)

    monkeypatch.setattr(kernel_build_cache, "_detect_arch_from_torch", lambda: "8.9")
    detected_l40s = kernel_build_cache._build_metadata(tmp_path)

    assert explicit_hopper["cache_key"] == detected_hopper["cache_key"]
    assert explicit_hopper["build"]["torch_cuda_arch_list"] == "9.0a"
    assert detected_hopper["build"]["torch_cuda_arch_list"] == ""
    assert detected_l40s["cache_key"] != detected_hopper["cache_key"]


def test_cache_key_changes_when_compiler_libc_metadata_changes(monkeypatch, tmp_path) -> None:
    _patch_stable_metadata(monkeypatch)
    monkeypatch.setattr(kernel_build_cache, "_detect_arch_from_torch", lambda: "9.0a")

    monkeypatch.setattr(
        kernel_build_cache,
        "_compiler_libc_metadata",
        lambda: {
            "compiler": {"gcc_version": "gcc 11.4.0"},
            "libc": {"ldd_version": "ldd 2.35"},
        },
    )
    gcc_11 = kernel_build_cache._build_metadata(tmp_path)

    monkeypatch.setattr(
        kernel_build_cache,
        "_compiler_libc_metadata",
        lambda: {
            "compiler": {"gcc_version": "gcc 12.3.0"},
            "libc": {"ldd_version": "ldd 2.36"},
        },
    )
    gcc_12 = kernel_build_cache._build_metadata(tmp_path)

    assert gcc_11["schema_version"] == 2
    assert gcc_11["cache_key"] != gcc_12["cache_key"]


def test_find_wheel_uses_fastvideo_kernel_wheel_patterns(tmp_path) -> None:
    (tmp_path / "unrelated-0.1.0.whl").write_text("")
    (tmp_path / "fastvideo_kernel-0.3.1-cp310-cp310-linux_x86_64.whl").write_text("")
    expected = tmp_path / "fastvideo_kernel-0.3.2-cp310-cp310-linux_x86_64.whl"
    expected.write_text("")

    assert kernel_build_cache._find_wheel(tmp_path) == expected


def _write_cache_entry(cache_root: Path, cache_key: str, wheel_name: str) -> Path:
    cache_entry = cache_root / cache_key
    cache_entry.mkdir()
    (cache_entry / wheel_name).write_text("cached wheel")
    (cache_entry / kernel_build_cache.METADATA_FILE).write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "schema_version": kernel_build_cache.CACHE_SCHEMA_VERSION,
            }))
    return cache_entry


def test_store_cache_entry_handles_concurrent_rename_error(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    wheel = tmp_path / "fastvideo_kernel-0.3.2-cp310-cp310-linux_x86_64.whl"
    wheel.write_text("wheel")
    metadata = {
        "cache_key": "cache-key",
        "schema_version": kernel_build_cache.CACHE_SCHEMA_VERSION,
    }
    rename_calls = []

    def fake_rename(self, target):
        rename_calls.append((self, target))
        _write_cache_entry(target.parent, target.name, wheel.name)
        raise OSError(errno.ENOTEMPTY, "Directory not empty")

    monkeypatch.setattr(Path, "rename", fake_rename)

    cache_entry = kernel_build_cache._store_cache_entry(cache_root, metadata, wheel)

    assert cache_entry == cache_root / "cache-key"
    assert rename_calls
    assert not list(cache_root.glob(".cache-key.tmp-*"))


def test_store_cache_entry_uses_unique_temp_dir_without_deleting_existing_collision(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    wheel = tmp_path / "fastvideo_kernel-0.3.2-cp310-cp310-linux_x86_64.whl"
    wheel.write_text("wheel")
    metadata = {
        "cache_key": "cache-key",
        "schema_version": kernel_build_cache.CACHE_SCHEMA_VERSION,
    }
    existing_temp = cache_root / ".cache-key.tmp-123"
    existing_temp.mkdir()
    sentinel = existing_temp / "sentinel"
    sentinel.write_text("owned by another writer")
    monkeypatch.setattr(kernel_build_cache.os, "getpid", lambda: 123)

    cache_entry = kernel_build_cache._store_cache_entry(cache_root, metadata, wheel)

    assert cache_entry == cache_root / "cache-key"
    assert sentinel.read_text() == "owned by another writer"
    assert kernel_build_cache._cache_hit(cache_entry, "cache-key") == cache_entry / wheel.name


def test_store_cache_entry_raises_when_renamed_entry_fails_validation(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    wheel = tmp_path / "fastvideo_kernel-0.3.2-cp310-cp310-linux_x86_64.whl"
    wheel.write_text("wheel")
    metadata = {
        "cache_key": "cache-key",
        "schema_version": kernel_build_cache.CACHE_SCHEMA_VERSION,
    }
    original_rename = Path.rename

    def fake_rename(self, target):
        original_rename(self, target)
        (target / kernel_build_cache.METADATA_FILE).write_text(
            json.dumps({
                "cache_key": "other-key",
                "schema_version": kernel_build_cache.CACHE_SCHEMA_VERSION,
            }))

    monkeypatch.setattr(Path, "rename", fake_rename)

    with pytest.raises(RuntimeError, match="could not be validated"):
        kernel_build_cache._store_cache_entry(cache_root, metadata, wheel)

    assert not (cache_root / "cache-key").exists()


def test_store_cache_entry_raises_when_rename_fails_without_valid_entry(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    wheel = tmp_path / "fastvideo_kernel-0.3.2-cp310-cp310-linux_x86_64.whl"
    wheel.write_text("wheel")
    metadata = {
        "cache_key": "cache-key",
        "schema_version": kernel_build_cache.CACHE_SCHEMA_VERSION,
    }

    def fake_rename(self, target):
        raise OSError(errno.ENOTEMPTY, "Directory not empty")

    monkeypatch.setattr(Path, "rename", fake_rename)

    with pytest.raises(RuntimeError, match="Failed to store kernel cache entry"):
        kernel_build_cache._store_cache_entry(cache_root, metadata, wheel)

    assert not list(cache_root.glob(".cache-key.tmp-*"))


def test_store_cache_entry_rejects_existing_invalid_entry(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    wheel = tmp_path / "fastvideo_kernel-0.3.2-cp310-cp310-linux_x86_64.whl"
    wheel.write_text("wheel")
    metadata = {
        "cache_key": "cache-key",
        "schema_version": kernel_build_cache.CACHE_SCHEMA_VERSION,
    }
    invalid_entry = cache_root / "cache-key"
    invalid_entry.mkdir()
    (invalid_entry / kernel_build_cache.METADATA_FILE).write_text(
        json.dumps({"cache_key": "other-key"}))

    with pytest.raises(RuntimeError, match="does not contain a valid wheel"):
        kernel_build_cache._store_cache_entry(cache_root, metadata, wheel)
