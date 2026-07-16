# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import importlib.util
import json
import subprocess
import zipfile
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
WHEEL_NAME = "fastvideo_kernel-0.3.2-py3-none-any.whl"


def _write_wheel(path: Path, *, distribution: str = "fastvideo-kernel", payload: str = "payload") -> Path:
    dist_info = "fastvideo_kernel-0.3.2.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fastvideo_kernel/__init__.py", payload)
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.3.2\n",
        )
    return path


def _metadata(cache_key: str = "cache-key") -> dict[str, object]:
    return {
        "cache_key": cache_key,
        "schema_version": kernel_build_cache.CACHE_SCHEMA_VERSION,
    }


def _write_cache_entry(cache_root: Path, cache_key: str, *, payload: str = "payload") -> Path:
    cache_entry = cache_root / cache_key
    cache_entry.mkdir()
    wheel = _write_wheel(cache_entry / WHEEL_NAME, payload=payload)
    stored_metadata = {
        **_metadata(cache_key),
        "artifact": kernel_build_cache._wheel_artifact(wheel),
    }
    (cache_entry / kernel_build_cache.METADATA_FILE).write_text(json.dumps(stored_metadata), encoding="utf-8")
    return cache_entry


def _patch_stable_metadata(monkeypatch) -> None:
    for name in (
        "GPU_BACKEND",
        "CMAKE_ARGS",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "TORCH_CUDA_ARCH_LIST",
        "FASTVIDEO_CONTAINER_IMAGE_REF",
        "CUDACXX",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(kernel_build_cache, "_kernel_source_hash", lambda repo_root: {"source_hash": "source"})
    monkeypatch.setattr(kernel_build_cache, "_read_kernel_version", lambda repo_root: "0.3.2")
    monkeypatch.setattr(
        kernel_build_cache,
        "_torch_metadata",
        lambda: {
            "torch_version": "2.9.0",
            "torch_cuda_version": "12.8",
            "torch_file": "/opt/venv/lib/python3.12/site-packages/torch/__init__.py",
            "torch_config": "USE_CUDA=ON",
            "cxx11_abi": True,
        },
    )
    monkeypatch.setattr(
        kernel_build_cache,
        "_compiler_libc_metadata",
        lambda: {
            "compiler": {"cc": {"path": "/usr/bin/gcc", "version": "gcc 11.4.0"}},
            "libc": {"ldd_version": "ldd 2.35"},
        },
    )
    monkeypatch.setattr(
        kernel_build_cache,
        "_selected_command_metadata",
        lambda environment_name, default_command: {
            "raw": default_command,
            "command": json.dumps([default_command]),
            "resolved_executable": f"/usr/bin/{default_command}",
            "version": f"{default_command} 12.8",
        },
    )


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


@pytest.mark.parametrize("environment_name", ["CFLAGS", "CXXFLAGS", "LDFLAGS"])
def test_cache_key_changes_with_build_flags(monkeypatch, tmp_path, environment_name) -> None:
    _patch_stable_metadata(monkeypatch)
    monkeypatch.setattr(kernel_build_cache, "_detect_arch_from_torch", lambda: "9.0a")
    baseline = kernel_build_cache._build_metadata(tmp_path)

    monkeypatch.setenv(environment_name, "-DFASTVIDEO_ABI_VARIANT=1")
    changed = kernel_build_cache._build_metadata(tmp_path)

    assert baseline["cache_key"] != changed["cache_key"]


def test_cache_key_changes_with_compiler_or_torch_abi(monkeypatch, tmp_path) -> None:
    _patch_stable_metadata(monkeypatch)
    monkeypatch.setattr(kernel_build_cache, "_detect_arch_from_torch", lambda: "9.0a")
    baseline = kernel_build_cache._build_metadata(tmp_path)

    monkeypatch.setattr(
        kernel_build_cache,
        "_compiler_libc_metadata",
        lambda: {
            "compiler": {"cc": {"path": "/opt/gcc-12/bin/gcc", "version": "gcc 12.3.0"}},
            "libc": {"ldd_version": "ldd 2.35"},
        },
    )
    compiler_changed = kernel_build_cache._build_metadata(tmp_path)

    _patch_stable_metadata(monkeypatch)
    torch_metadata = kernel_build_cache._torch_metadata()
    monkeypatch.setattr(kernel_build_cache, "_torch_metadata", lambda: {**torch_metadata, "cxx11_abi": False})
    torch_abi_changed = kernel_build_cache._build_metadata(tmp_path)

    assert baseline["schema_version"] == 3
    assert baseline["cache_key"] != compiler_changed["cache_key"]
    assert baseline["cache_key"] != torch_abi_changed["cache_key"]


def test_find_wheel_uses_fastvideo_kernel_wheel_patterns(tmp_path) -> None:
    (tmp_path / "unrelated-0.1.0.whl").write_text("")
    (tmp_path / "fastvideo_kernel-0.3.1-py3-none-any.whl").write_text("")
    expected = tmp_path / WHEEL_NAME
    expected.write_text("")

    assert kernel_build_cache._find_wheel(tmp_path) == expected


def test_wheel_artifact_validates_archive_digest_size_and_distribution(tmp_path) -> None:
    wheel = _write_wheel(tmp_path / WHEEL_NAME)
    artifact = kernel_build_cache._wheel_artifact(wheel)

    valid, reason = kernel_build_cache._validate_wheel(wheel, artifact)

    assert valid
    assert reason == ""
    assert artifact["distribution"] == "fastvideo-kernel"
    assert artifact["size_bytes"] == wheel.stat().st_size
    assert len(artifact["sha256"]) == 64


@pytest.mark.parametrize("corruption", ["digest", "size", "archive", "distribution"])
def test_wheel_validation_rejects_corrupt_or_unexpected_artifacts(tmp_path, corruption) -> None:
    wheel = _write_wheel(tmp_path / WHEEL_NAME)
    artifact = kernel_build_cache._wheel_artifact(wheel)
    if corruption == "digest":
        artifact["sha256"] = "0" * 64
    elif corruption == "size":
        artifact["size_bytes"] = int(artifact["size_bytes"]) + 1
    elif corruption == "archive":
        wheel.write_bytes(b"not a wheel")
    else:
        wheel = _write_wheel(tmp_path / "unexpected.whl", distribution="other-package")
        artifact = {**artifact, "wheel_name": wheel.name}

    valid, reason = kernel_build_cache._validate_wheel(wheel, artifact)

    assert not valid
    assert reason


def test_store_cache_entry_handles_concurrent_valid_entry(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    wheel = _write_wheel(tmp_path / WHEEL_NAME)
    rename_calls = []

    def fake_rename(self, target):
        rename_calls.append((self, target))
        _write_cache_entry(target.parent, target.name)
        raise OSError(errno.ENOTEMPTY, "Directory not empty")

    monkeypatch.setattr(Path, "rename", fake_rename)

    cache_entry = kernel_build_cache._store_cache_entry(cache_root, _metadata(), wheel)

    assert cache_entry == cache_root / "cache-key"
    assert rename_calls
    assert not list(cache_root.glob(".cache-key.tmp-*"))


def test_store_cache_entry_evicts_existing_invalid_entry(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    invalid_entry = cache_root / "cache-key"
    invalid_entry.mkdir()
    sentinel = invalid_entry / "stale"
    sentinel.write_text("invalid")
    wheel = _write_wheel(tmp_path / WHEEL_NAME)

    cache_entry = kernel_build_cache._store_cache_entry(cache_root, _metadata(), wheel)
    cached_wheel, reason = kernel_build_cache._cache_entry_wheel(cache_entry, "cache-key")

    assert cached_wheel == cache_entry / WHEEL_NAME
    assert reason == ""
    assert not sentinel.exists()


def test_store_cache_entry_raises_when_renamed_entry_fails_validation(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    wheel = _write_wheel(tmp_path / WHEEL_NAME)
    original_rename = Path.rename

    def fake_rename(self, target):
        original_rename(self, target)
        metadata_path = target / kernel_build_cache.METADATA_FILE
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["cache_key"] = "other-key"
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(Path, "rename", fake_rename)

    with pytest.raises(RuntimeError, match="could not be validated"):
        kernel_build_cache._store_cache_entry(cache_root, _metadata(), wheel)

    assert not (cache_root / "cache-key").exists()


def test_consumer_installs_valid_cache_hit_without_build(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_entry = _write_cache_entry(cache_root, "cache-key")
    installed = []
    monkeypatch.setattr(kernel_build_cache, "_build_metadata", lambda repo_root: _metadata())
    monkeypatch.setattr(kernel_build_cache, "_try_install_prebuilt", lambda metadata, path: False)
    monkeypatch.setattr(kernel_build_cache, "_install_wheel", installed.append)
    monkeypatch.setattr(
        kernel_build_cache,
        "_build_and_install_local",
        lambda repo_root, metadata: pytest.fail("cache hit unexpectedly built locally"),
    )

    kernel_build_cache.install_cached_or_build(tmp_path, cache_root, tmp_path / "missing-prebuilt.json")

    assert installed == [cache_entry / WHEEL_NAME]


def test_consumer_rejects_corrupt_cache_without_mutating_it(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_entry = _write_cache_entry(cache_root, "cache-key")
    wheel = cache_entry / WHEEL_NAME
    wheel.write_bytes(b"corrupt")
    before_metadata = (cache_entry / kernel_build_cache.METADATA_FILE).read_bytes()
    built = []
    monkeypatch.setattr(kernel_build_cache, "_build_metadata", lambda repo_root: _metadata())
    monkeypatch.setattr(kernel_build_cache, "_try_install_prebuilt", lambda metadata, path: False)
    monkeypatch.setattr(kernel_build_cache, "_build_and_install_local", lambda repo_root, metadata: built.append(True))

    kernel_build_cache.install_cached_or_build(tmp_path, cache_root, tmp_path / "missing-prebuilt.json")

    assert built == [True]
    assert wheel.read_bytes() == b"corrupt"
    assert (cache_entry / kernel_build_cache.METADATA_FILE).read_bytes() == before_metadata


def test_consumer_falls_back_when_cached_install_fails(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    _write_cache_entry(cache_root, "cache-key")
    built = []
    monkeypatch.setattr(kernel_build_cache, "_build_metadata", lambda repo_root: _metadata())
    monkeypatch.setattr(kernel_build_cache, "_try_install_prebuilt", lambda metadata, path: False)
    monkeypatch.setattr(
        kernel_build_cache,
        "_install_wheel",
        lambda wheel: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["uv"])),
    )
    monkeypatch.setattr(kernel_build_cache, "_build_and_install_local", lambda repo_root, metadata: built.append(True))

    kernel_build_cache.install_cached_or_build(tmp_path, cache_root, tmp_path / "missing-prebuilt.json")

    assert built == [True]


def test_prebuilt_requires_validated_artifact(monkeypatch, tmp_path) -> None:
    wheel = _write_wheel(tmp_path / WHEEL_NAME)
    prebuilt_info = tmp_path / "prebuilt.json"
    payload = {
        **_metadata(),
        "artifact": kernel_build_cache._wheel_artifact(wheel),
        "wheel_path": str(wheel),
    }
    prebuilt_info.write_text(json.dumps(payload), encoding="utf-8")
    installed = []
    monkeypatch.setattr(kernel_build_cache, "_install_wheel", installed.append)

    assert kernel_build_cache._try_install_prebuilt(_metadata(), prebuilt_info)
    assert installed == [wheel]

    installed.clear()
    wheel.write_bytes(b"corrupt")
    assert not kernel_build_cache._try_install_prebuilt(_metadata(), prebuilt_info)
    assert installed == []


def test_producer_repairs_invalid_entry_and_stores_valid_wheel(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    invalid_entry = cache_root / "cache-key"
    invalid_entry.mkdir()
    (invalid_entry / kernel_build_cache.METADATA_FILE).write_text("not json", encoding="utf-8")
    built_dir = tmp_path / "built"
    built_dir.mkdir()
    built_wheel = _write_wheel(built_dir / WHEEL_NAME)
    installed = []
    monkeypatch.setattr(kernel_build_cache, "_build_metadata", lambda repo_root: _metadata())
    monkeypatch.setattr(kernel_build_cache, "_build_wheel", lambda repo_root, metadata: built_wheel)
    monkeypatch.setattr(kernel_build_cache, "_install_wheel", installed.append)

    kernel_build_cache.produce_cached_wheel(tmp_path, cache_root)

    cached_wheel, reason = kernel_build_cache._cache_entry_wheel(cache_root / "cache-key", "cache-key")
    assert installed == [built_wheel]
    assert cached_wheel == cache_root / "cache-key" / WHEEL_NAME
    assert reason == ""
