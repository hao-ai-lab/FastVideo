"""Build and reuse FastVideo kernel wheels in Modal CI.

This module stays standalone because Modal launchers execute it from a freshly
cloned checkout after dependency installation.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_ROOT = "/root/fastvideo-kernel-cache"
DEFAULT_PREBUILT_INFO_PATH = "/opt/fastvideo-kernel-build-info.json"
KERNEL_RELATIVE_DIR = "fastvideo-kernel"
METADATA_FILE = "metadata.json"


def _utc_now_isoformat() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _log(message: str) -> None:
    print(f"[fastvideo-kernel-cache] {message}", flush=True)


def _run(args: list[str],
         *,
         cwd: Path | None = None,
         env: dict[str, str] | None = None,
         capture: bool = False) -> str:
    _log("$ " + " ".join(str(arg) for arg in args))
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture:
        return result.stdout.strip()
    return ""


def _run_optional(args: list[str], *, cwd: Path | None = None) -> str:
    try:
        return _run(args, cwd=cwd, capture=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return f"<unavailable: {error}>"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _hash_directory(root: Path) -> str:
    hasher = hashlib.sha256()
    skip_dirs = {".git", ".mypy_cache", ".pytest_cache", "__pycache__", "build", "dist"}
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            dirname for dirname in dirnames if dirname not in skip_dirs and not dirname.endswith(".egg-info"))
        for filename in sorted(filenames):
            path = Path(current_root) / filename
            hasher.update(path.relative_to(root).as_posix().encode("utf-8"))
            hasher.update(b"\0")
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    hasher.update(chunk)
            hasher.update(b"\0")
    return hasher.hexdigest()


def _kernel_source_hash(repo_root: Path) -> dict[str, str]:
    kernel_root = repo_root / KERNEL_RELATIVE_DIR
    try:
        tree_hash = _run(["git", "rev-parse", f"HEAD:{KERNEL_RELATIVE_DIR}"], cwd=repo_root, capture=True)
        diff = _run_optional(["git", "diff", "--binary", "--", KERNEL_RELATIVE_DIR], cwd=repo_root)
        cached_diff = _run_optional(["git", "diff", "--cached", "--binary", "--", KERNEL_RELATIVE_DIR],
                                    cwd=repo_root)
        submodules = _run_optional(
            [
                "git",
                "submodule",
                "status",
                "--",
                "fastvideo-kernel/include/cutlass",
                "fastvideo-kernel/include/tk",
            ],
            cwd=repo_root,
        )
        descriptor = {
            "mode": "git",
            "tree_hash": tree_hash,
            "diff_hash": _sha256_text(diff) if diff else "",
            "cached_diff_hash": _sha256_text(cached_diff) if cached_diff else "",
            "submodules_hash": _sha256_text(submodules),
        }
        descriptor["source_hash"] = _sha256_text(json.dumps(descriptor, sort_keys=True))
        return descriptor
    except (FileNotFoundError, subprocess.CalledProcessError):
        descriptor = {
            "mode": "directory",
            "tree_hash": _hash_directory(kernel_root),
            "diff_hash": "",
            "cached_diff_hash": "",
            "submodules_hash": "",
        }
        descriptor["source_hash"] = descriptor["tree_hash"]
        return descriptor


def _read_kernel_version(repo_root: Path) -> str:
    pyproject = repo_root / KERNEL_RELATIVE_DIR / "pyproject.toml"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("version"):
            _, _, value = stripped.partition("=")
            return value.strip().strip("\"'")
    return "<unknown>"


def _detect_arch_from_torch() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "<unavailable>"
        major, minor = torch.cuda.get_device_capability(0)
        if major == 9 and minor == 0:
            return "9.0a"
        if major == 12 and minor == 0:
            return "12.0a"
        return f"{major}.{minor}"
    except Exception as error:  # noqa: BLE001 - metadata should explain fallback
        return f"<unavailable: {error}>"


def _torch_metadata() -> dict[str, str]:
    try:
        import torch

        return {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "torch_file": str(getattr(torch, "__file__", "")),
        }
    except Exception as error:  # noqa: BLE001 - report in metadata
        return {
            "torch_version": f"<unavailable: {error}>",
            "torch_cuda_version": "<unavailable>",
            "torch_file": "<unavailable>",
        }


def _build_metadata(repo_root: Path) -> dict[str, object]:
    explicit_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    resolved_arch = explicit_arch or _detect_arch_from_torch()
    cache_key_build = {
        "gpu_backend": os.environ.get("GPU_BACKEND", "CUDA"),
        "resolved_torch_cuda_arch_list": resolved_arch,
        "cmake_args": " ".join(os.environ.get("CMAKE_ARGS", "").split()),
    }
    cache_key_metadata: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kernel_version": _read_kernel_version(repo_root),
        "source": _kernel_source_hash(repo_root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "platform": sysconfig.get_platform(),
            "machine": platform.machine(),
        },
        "torch": _torch_metadata(),
        "cuda": {
            "cuda_home": os.environ.get("CUDA_HOME", ""),
            "cudacxx": os.environ.get("CUDACXX", ""),
            "nvcc_version": _run_optional(["nvcc", "--version"]),
        },
        "build": cache_key_build,
    }
    metadata = {
        **cache_key_metadata,
        "build": {
            **cache_key_build,
            "torch_cuda_arch_list": explicit_arch,
        },
    }
    metadata["cache_key"] = _sha256_text(json.dumps(cache_key_metadata, sort_keys=True, separators=(",", ":")))
    return metadata


def _find_wheel(directory: Path) -> Path:
    candidates = sorted(
        path
        for pattern in ("fastvideo_kernel-*.whl", "fastvideo-kernel-*.whl")
        for path in directory.glob(pattern)
    )
    if not candidates:
        raise RuntimeError(f"No fastvideo-kernel wheel found in {directory}")
    return candidates[-1]


def _install_wheel(wheel: Path) -> None:
    _log(f"installing wheel {wheel}")
    _run([
        "uv",
        "pip",
        "install",
        str(wheel),
        "--reinstall-package",
        "fastvideo-kernel",
        "--no-deps",
    ])


def _metadata_matches(path: Path, cache_key: str) -> bool:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("cache_key") == cache_key and metadata.get("schema_version") == CACHE_SCHEMA_VERSION


def _try_install_prebuilt(metadata: dict[str, object], prebuilt_info_path: Path) -> bool:
    cache_key = str(metadata["cache_key"])
    if not prebuilt_info_path.exists():
        return False
    if not _metadata_matches(prebuilt_info_path, cache_key):
        _log(f"Docker-prebuilt kernel metadata mismatch at {prebuilt_info_path}; using Modal cache")
        return False
    prebuilt = json.loads(prebuilt_info_path.read_text(encoding="utf-8"))
    wheel_path = Path(str(prebuilt.get("wheel_path", "")))
    if not wheel_path.is_file():
        _log(f"Docker-prebuilt metadata matched but wheel is missing at {wheel_path}; using Modal cache")
        return False
    _log(f"using Docker-prebuilt kernel wheel for cache key {cache_key}")
    _install_wheel(wheel_path)
    return True


def _cache_hit(cache_entry: Path, cache_key: str) -> Path | None:
    if not _metadata_matches(cache_entry / METADATA_FILE, cache_key):
        return None
    try:
        return _find_wheel(cache_entry)
    except (OSError, RuntimeError):
        return None


def _store_cache_entry(cache_root: Path, metadata: dict[str, object], wheel: Path) -> Path:
    cache_key = str(metadata["cache_key"])
    cache_entry = cache_root / cache_key
    if cache_entry.exists():
        if _cache_hit(cache_entry, cache_key) is not None:
            _log(f"cache entry {cache_entry} appeared after build; leaving existing entry in place")
            return cache_entry
        raise RuntimeError(
            f"Cache entry {cache_entry} exists but does not contain a valid wheel for this build")

    temp_entry = cache_root / f".{cache_key}.tmp-{os.getpid()}"
    if temp_entry.exists():
        shutil.rmtree(temp_entry)
    temp_entry.mkdir(parents=True)
    shutil.copy2(wheel, temp_entry / wheel.name)
    stored_metadata = {
        **metadata,
        "created_at_utc": _utc_now_isoformat(),
        "wheel_name": wheel.name,
    }
    (temp_entry / METADATA_FILE).write_text(json.dumps(stored_metadata, indent=2, sort_keys=True),
                                            encoding="utf-8")
    try:
        temp_entry.rename(cache_entry)
    except OSError as error:
        shutil.rmtree(temp_entry, ignore_errors=True)
        if _cache_hit(cache_entry, cache_key) is not None:
            _log(f"cache entry {cache_entry} was created concurrently; leaving existing entry in place")
            return cache_entry
        raise RuntimeError(f"Failed to store kernel cache entry at {cache_entry}") from error
    return cache_entry


def _build_wheel(repo_root: Path, metadata: dict[str, object]) -> Path:
    with tempfile.TemporaryDirectory(prefix="fastvideo-kernel-wheel-") as temp_dir:
        wheel_dir = Path(temp_dir)
        env = os.environ.copy()
        resolved_arch = str(metadata["build"]["resolved_torch_cuda_arch_list"])  # type: ignore[index]
        if not env.get("TORCH_CUDA_ARCH_LIST") and not resolved_arch.startswith("<unavailable"):
            env["TORCH_CUDA_ARCH_LIST"] = resolved_arch
        _run(["./build.sh", "--wheel-dir", str(wheel_dir)], cwd=repo_root / KERNEL_RELATIVE_DIR, env=env)
        wheel = _find_wheel(wheel_dir)
        persisted = Path(tempfile.mkdtemp(prefix="fastvideo-kernel-built-wheel-")) / wheel.name
        shutil.copy2(wheel, persisted)
        return persisted


def install_cached_or_build(repo_root: Path, cache_root: Path, prebuilt_info_path: Path) -> None:
    metadata = _build_metadata(repo_root)
    cache_key = str(metadata["cache_key"])
    _log(f"resolved cache key {cache_key}")

    if _try_install_prebuilt(metadata, prebuilt_info_path):
        return

    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _log(f"cache unavailable at {cache_root}: {error}; building from source without storing")
        wheel = _build_wheel(repo_root, metadata)
        try:
            _install_wheel(wheel)
        finally:
            shutil.rmtree(wheel.parent, ignore_errors=True)
        return

    cache_entry = cache_root / cache_key
    cached_wheel = _cache_hit(cache_entry, cache_key)
    if cached_wheel is not None:
        _log(f"cache hit: {cached_wheel}")
        _install_wheel(cached_wheel)
        return

    _log(f"cache miss: {cache_entry}")
    wheel = _build_wheel(repo_root, metadata)
    try:
        _install_wheel(wheel)
        stored_entry = _store_cache_entry(cache_root, metadata, wheel)
        _log(f"stored wheel cache entry: {stored_entry}")
    finally:
        shutil.rmtree(wheel.parent, ignore_errors=True)


def write_build_info(repo_root: Path, wheel_dir: Path, output: Path) -> None:
    metadata = _build_metadata(repo_root)
    wheel = _find_wheel(wheel_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "created_at_utc": _utc_now_isoformat(),
        "wheel_name": wheel.name,
        "wheel_path": str(wheel),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"wrote Docker-prebuilt kernel metadata: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "write-build-info", "print-key"))
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--cache-root", default=os.environ.get("FASTVIDEO_KERNEL_CACHE_ROOT", DEFAULT_CACHE_ROOT))
    parser.add_argument("--prebuilt-info-path",
                        default=os.environ.get("FASTVIDEO_KERNEL_PREBUILT_INFO", DEFAULT_PREBUILT_INFO_PATH))
    parser.add_argument("--wheel-dir", default="")
    parser.add_argument("--output", default=DEFAULT_PREBUILT_INFO_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    if args.command == "install":
        install_cached_or_build(repo_root, Path(args.cache_root), Path(args.prebuilt_info_path))
    elif args.command == "write-build-info":
        if not args.wheel_dir:
            raise RuntimeError("--wheel-dir is required for write-build-info")
        write_build_info(repo_root, Path(args.wheel_dir), Path(args.output))
    elif args.command == "print-key":
        print(json.dumps(_build_metadata(repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
