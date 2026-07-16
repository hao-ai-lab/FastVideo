"""Build and reuse FastVideo kernel wheels in Modal CI.

This module stays standalone because Modal launchers execute it from a freshly
cloned checkout after dependency installation. Shared cache consumers are
read-only; only the explicit ``produce`` command may mutate cache state.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path


CACHE_SCHEMA_VERSION = 3
DEFAULT_PREBUILT_INFO_PATH = "/opt/fastvideo-kernel-build-info.json"
KERNEL_RELATIVE_DIR = "fastvideo-kernel"
METADATA_FILE = "metadata.json"
EXPECTED_DISTRIBUTION = "fastvideo-kernel"


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
    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as error:
        return f"<unavailable: {error}>"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def _torch_metadata() -> dict[str, object]:
    try:
        import torch

        compiled_with_cxx11_abi = getattr(torch, "compiled_with_cxx11_abi", None)
        if callable(compiled_with_cxx11_abi):
            cxx11_abi: object = bool(compiled_with_cxx11_abi())
        else:
            cxx11_abi = getattr(getattr(torch, "_C", object()), "_GLIBCXX_USE_CXX11_ABI", "<unavailable>")
        return {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "torch_file": str(getattr(torch, "__file__", "")),
            "torch_config": str(torch.__config__.show()),
            "cxx11_abi": cxx11_abi,
        }
    except Exception as error:  # noqa: BLE001 - report in metadata
        return {
            "torch_version": f"<unavailable: {error}>",
            "torch_cuda_version": "<unavailable>",
            "torch_file": "<unavailable>",
            "torch_config": "<unavailable>",
            "cxx11_abi": "<unavailable>",
        }


def _selected_command_metadata(environment_name: str, default_command: str) -> dict[str, str]:
    raw = os.environ.get(environment_name, "").strip() or default_command
    try:
        command = shlex.split(raw)
    except ValueError as error:
        return {
            "raw": raw,
            "command": "<invalid>",
            "resolved_executable": "<invalid>",
            "version": f"<unavailable: {error}>",
        }
    if not command:
        command = [default_command]
    resolved_executable = shutil.which(command[0]) or command[0]
    return {
        "raw": raw,
        "command": json.dumps(command),
        "resolved_executable": resolved_executable,
        "version": _run_optional([*command, "--version"]),
    }


def _compiler_libc_metadata() -> dict[str, object]:
    return {
        "compiler": {
            "cc": _selected_command_metadata("CC", "cc"),
            "cxx": _selected_command_metadata("CXX", "c++"),
        },
        "build_tools": {
            "cmake_version": _run_optional(["cmake", "--version"]),
            "ninja_version": _run_optional(["ninja", "--version"]),
            "linker": _selected_command_metadata("LD", "ld"),
        },
        "libc": {
            "platform_libc": list(platform.libc_ver()),
            "ldd_version": _run_optional(["ldd", "--version"]),
        },
    }


def _build_metadata(repo_root: Path) -> dict[str, object]:
    explicit_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    resolved_arch = explicit_arch or _detect_arch_from_torch()
    cache_key_build = {
        "gpu_backend": os.environ.get("GPU_BACKEND", "CUDA"),
        "resolved_torch_cuda_arch_list": resolved_arch,
        "cmake_args": os.environ.get("CMAKE_ARGS", ""),
        "cflags": os.environ.get("CFLAGS", ""),
        "cxxflags": os.environ.get("CXXFLAGS", ""),
        "ldflags": os.environ.get("LDFLAGS", ""),
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
            "nvcc": _selected_command_metadata("CUDACXX", "nvcc"),
        },
        "abi": _compiler_libc_metadata(),
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


def _normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_artifact(wheel: Path) -> dict[str, object]:
    if not wheel.is_file() or wheel.is_symlink():
        raise RuntimeError(f"wheel is missing or is not a regular file: {wheel}")
    try:
        with zipfile.ZipFile(wheel) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"wheel archive has a corrupt member: {bad_member}")
            names = archive.namelist()
            wheel_metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
            package_metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(wheel_metadata) != 1 or len(package_metadata) != 1:
                raise RuntimeError("wheel archive must contain one WHEEL and one METADATA file")
            parsed_metadata = Parser().parsestr(archive.read(package_metadata[0]).decode("utf-8"))
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"invalid wheel archive: {error}") from error

    distribution = _normalize_distribution(parsed_metadata.get("Name", ""))
    if distribution != EXPECTED_DISTRIBUTION:
        raise RuntimeError(f"unexpected wheel distribution {distribution!r}")
    return {
        "wheel_name": wheel.name,
        "size_bytes": wheel.stat().st_size,
        "sha256": _sha256_file(wheel),
        "distribution": distribution,
    }


def _validate_wheel(wheel: Path, expected_artifact: object) -> tuple[bool, str]:
    if not isinstance(expected_artifact, dict):
        return False, "artifact metadata is missing"
    try:
        actual_artifact = _wheel_artifact(wheel)
    except RuntimeError as error:
        return False, str(error)
    for field in ("wheel_name", "size_bytes", "sha256", "distribution"):
        if actual_artifact[field] != expected_artifact.get(field):
            return False, f"artifact {field} mismatch"
    return True, ""


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


def _load_metadata(path: Path) -> tuple[dict[str, object] | None, str]:
    if path.is_symlink():
        return None, "metadata is a symlink"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"metadata is unreadable: {error}"
    if not isinstance(metadata, dict):
        return None, "metadata is not an object"
    return metadata, ""


def _validated_payload_wheel(payload: dict[str, object], wheel: Path, cache_key: str) -> tuple[Path | None, str]:
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None, "schema version mismatch"
    if payload.get("cache_key") != cache_key:
        return None, "cache key mismatch"
    valid, reason = _validate_wheel(wheel, payload.get("artifact"))
    if not valid:
        return None, reason
    return wheel, ""


def _cache_entry_wheel(cache_entry: Path, cache_key: str) -> tuple[Path | None, str]:
    if cache_entry.is_symlink():
        return None, "cache entry is a symlink"
    metadata, reason = _load_metadata(cache_entry / METADATA_FILE)
    if metadata is None:
        return None, reason
    artifact = metadata.get("artifact")
    wheel_name = artifact.get("wheel_name") if isinstance(artifact, dict) else None
    if not isinstance(wheel_name, str):
        return None, "artifact wheel name is missing"
    wheel = cache_entry / wheel_name
    return _validated_payload_wheel(metadata, wheel, cache_key)


def _try_install_prebuilt(metadata: dict[str, object], prebuilt_info_path: Path) -> bool:
    cache_key = str(metadata["cache_key"])
    if not prebuilt_info_path.exists():
        return False
    prebuilt, reason = _load_metadata(prebuilt_info_path)
    if prebuilt is None:
        _log(f"Docker-prebuilt kernel metadata is invalid at {prebuilt_info_path}: {reason}")
        return False
    wheel_path = Path(str(prebuilt.get("wheel_path", "")))
    wheel, reason = _validated_payload_wheel(prebuilt, wheel_path, cache_key)
    if wheel is None:
        _log(f"Docker-prebuilt kernel artifact rejected at {prebuilt_info_path}: {reason}")
        return False
    try:
        _log(f"using Docker-prebuilt kernel wheel for cache key {cache_key}")
        _install_wheel(wheel)
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as error:
        _log(f"Docker-prebuilt kernel installation failed: {error}; falling back")
        return False
    return True


def _store_cache_entry(cache_root: Path, metadata: dict[str, object], wheel: Path) -> Path:
    cache_key = str(metadata["cache_key"])
    cache_entry = cache_root / cache_key
    if cache_entry.exists() or cache_entry.is_symlink():
        cached_wheel, reason = _cache_entry_wheel(cache_entry, cache_key)
        if cached_wheel is not None:
            _log(f"cache entry {cache_entry} appeared after build; leaving existing entry in place")
            return cache_entry
        _log(f"evicting invalid cache entry {cache_entry}: {reason}")
        if cache_entry.is_dir() and not cache_entry.is_symlink():
            shutil.rmtree(cache_entry)
        else:
            cache_entry.unlink()

    temp_entry = Path(tempfile.mkdtemp(prefix=f".{cache_key}.tmp-", dir=cache_root))
    try:
        stored_wheel = temp_entry / wheel.name
        shutil.copy2(wheel, stored_wheel)
        stored_metadata = {
            **metadata,
            "created_at_utc": _utc_now_isoformat(),
            "artifact": _wheel_artifact(stored_wheel),
        }
        (temp_entry / METADATA_FILE).write_text(json.dumps(stored_metadata, indent=2, sort_keys=True),
                                                encoding="utf-8")
        temp_entry.rename(cache_entry)
    except OSError as error:
        shutil.rmtree(temp_entry, ignore_errors=True)
        cached_wheel, _ = _cache_entry_wheel(cache_entry, cache_key)
        if cached_wheel is not None:
            _log(f"cache entry {cache_entry} was created concurrently; leaving existing entry in place")
            return cache_entry
        raise RuntimeError(f"Failed to store kernel cache entry at {cache_entry}") from error

    cached_wheel, reason = _cache_entry_wheel(cache_entry, cache_key)
    if cached_wheel is None:
        shutil.rmtree(cache_entry, ignore_errors=True)
        raise RuntimeError(f"Stored kernel cache entry {cache_entry} could not be validated: {reason}")
    return cache_entry


def _build_wheel(repo_root: Path, metadata: dict[str, object]) -> Path:
    with tempfile.TemporaryDirectory(prefix="fastvideo-kernel-wheel-") as temp_dir:
        wheel_dir = Path(temp_dir)
        env = os.environ.copy()
        build_metadata = metadata["build"]
        if not isinstance(build_metadata, dict):
            raise RuntimeError("build metadata is invalid")
        resolved_arch = str(build_metadata["resolved_torch_cuda_arch_list"])
        if not env.get("TORCH_CUDA_ARCH_LIST") and not resolved_arch.startswith("<unavailable"):
            env["TORCH_CUDA_ARCH_LIST"] = resolved_arch
        _run(["./build.sh", "--wheel-dir", str(wheel_dir)], cwd=repo_root / KERNEL_RELATIVE_DIR, env=env)
        wheel = _find_wheel(wheel_dir)
        persisted = Path(tempfile.mkdtemp(prefix="fastvideo-kernel-built-wheel-")) / wheel.name
        shutil.copy2(wheel, persisted)
        return persisted


def _build_and_install_local(repo_root: Path, metadata: dict[str, object]) -> None:
    wheel = _build_wheel(repo_root, metadata)
    try:
        _wheel_artifact(wheel)
        _install_wheel(wheel)
    finally:
        shutil.rmtree(wheel.parent, ignore_errors=True)


def install_cached_or_build(repo_root: Path, cache_root: Path | None, prebuilt_info_path: Path) -> None:
    """Install from a validated artifact or build locally without writing shared state."""
    metadata = _build_metadata(repo_root)
    cache_key = str(metadata["cache_key"])
    _log(f"resolved cache key {cache_key}")

    if _try_install_prebuilt(metadata, prebuilt_info_path):
        return

    if cache_root is not None:
        cache_entry = cache_root / cache_key
        cached_wheel, reason = _cache_entry_wheel(cache_entry, cache_key)
        if cached_wheel is not None:
            try:
                _log(f"cache hit: {cached_wheel}")
                _install_wheel(cached_wheel)
                return
            except (FileNotFoundError, OSError, subprocess.CalledProcessError) as error:
                _log(f"cached wheel installation failed: {error}; building locally")
        else:
            _log(f"cache miss: {cache_entry} ({reason})")
    else:
        _log("shared cache disabled; using Docker-prebuilt artifact or a job-local build")

    _build_and_install_local(repo_root, metadata)


def produce_cached_wheel(repo_root: Path, cache_root: Path) -> None:
    """Build and store one validated cache entry from a trusted producer."""
    metadata = _build_metadata(repo_root)
    cache_key = str(metadata["cache_key"])
    _log(f"resolved cache key {cache_key}")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_entry = cache_root / cache_key
    cached_wheel, reason = _cache_entry_wheel(cache_entry, cache_key)
    if cached_wheel is not None:
        _log(f"cache hit: {cached_wheel}")
        return
    if cache_entry.exists() or cache_entry.is_symlink():
        _log(f"producer will repair invalid cache entry {cache_entry}: {reason}")

    _log(f"cache miss: {cache_entry}")
    wheel = _build_wheel(repo_root, metadata)
    try:
        _wheel_artifact(wheel)
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
        "artifact": _wheel_artifact(wheel),
        "wheel_path": str(wheel),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"wrote Docker-prebuilt kernel metadata: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "produce", "write-build-info", "print-key"))
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--cache-root", default=os.environ.get("FASTVIDEO_KERNEL_CACHE_ROOT", ""))
    parser.add_argument("--prebuilt-info-path",
                        default=os.environ.get("FASTVIDEO_KERNEL_PREBUILT_INFO", DEFAULT_PREBUILT_INFO_PATH))
    parser.add_argument("--wheel-dir", default="")
    parser.add_argument("--output", default=DEFAULT_PREBUILT_INFO_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    cache_root = Path(args.cache_root) if args.cache_root else None
    if args.command == "install":
        install_cached_or_build(repo_root, cache_root, Path(args.prebuilt_info_path))
    elif args.command == "produce":
        if cache_root is None:
            raise RuntimeError("--cache-root is required for produce")
        produce_cached_wheel(repo_root, cache_root)
    elif args.command == "write-build-info":
        if not args.wheel_dir:
            raise RuntimeError("--wheel-dir is required for write-build-info")
        write_build_info(repo_root, Path(args.wheel_dir), Path(args.output))
    elif args.command == "print-key":
        print(json.dumps(_build_metadata(repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
