# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import importlib.util
from pathlib import Path


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


def test_find_wheel_uses_fastvideo_kernel_wheel_patterns(tmp_path) -> None:
    (tmp_path / "unrelated-0.1.0.whl").write_text("")
    (tmp_path / "fastvideo_kernel-0.3.1-cp310-cp310-linux_x86_64.whl").write_text("")
    expected = tmp_path / "fastvideo_kernel-0.3.2-cp310-cp310-linux_x86_64.whl"
    expected.write_text("")

    assert kernel_build_cache._find_wheel(tmp_path) == expected


def test_store_cache_entry_handles_concurrent_rename_error(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    wheel = tmp_path / "fastvideo_kernel-0.3.2-cp310-cp310-linux_x86_64.whl"
    wheel.write_text("wheel")
    metadata = {"cache_key": "cache-key"}
    rename_calls = []

    def fake_rename(self, target):
        rename_calls.append((self, target))
        raise OSError(errno.ENOTEMPTY, "Directory not empty")

    monkeypatch.setattr(Path, "rename", fake_rename)

    cache_entry = kernel_build_cache._store_cache_entry(cache_root, metadata, wheel)

    assert cache_entry == cache_root / "cache-key"
    assert rename_calls
    assert not list(cache_root.glob(".cache-key.tmp-*"))
