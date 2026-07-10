"""Cross-container smoke for the Modal fastvideo-kernel build cache.

Example:
    python -m modal run fastvideo/tests/modal/kernel_cache_smoke.py \
        --git-repo https://github.com/macthecadillac/FastVideo.git \
        --git-commit <commit>

This is intentionally opt-in: it builds fastvideo-kernel on a real Modal GPU.
"""

import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from modal_image_utils import (  # noqa: E402
        resolve_image_ref, resolve_uv_torch_backend)
except ModuleNotFoundError:
    # Remote Modal containers re-import this module but mount only the
    # entrypoint file; the digest resolution already happened at local launch.
    def resolve_image_ref(image_ref: str) -> str:
        return image_ref

    def resolve_uv_torch_backend(image_tag: str) -> str | None:
        return os.environ.get("UV_TORCH_BACKEND")


app = modal.App("fastvideo-kernel-cache-smoke")

REPO_DIR = "/FastVideo"
KERNEL_CACHE_VOLUME_PATH = "/root/fastvideo-kernel-cache"
KERNEL_CACHE_VOLUME_NAME = "fastvideo-kernel-build-cache"
IMAGE_VERSION = os.getenv("IMAGE_VERSION", "latest")
IMAGE_TAG = os.environ.get(
    "FASTVIDEO_MODAL_IMAGE",
    f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{IMAGE_VERSION}",
)
IMAGE_REF = resolve_image_ref(IMAGE_TAG)

print(f"Using image: {IMAGE_REF}")
print(f"Using kernel cache volume: {KERNEL_CACHE_VOLUME_NAME}")

kernel_cache_vol = modal.Volume.from_name(KERNEL_CACHE_VOLUME_NAME, create_if_missing=True)
uv_torch_backend_override = resolve_uv_torch_backend(IMAGE_TAG)

image = (
    modal.Image.from_registry(IMAGE_REF, add_python="3.12")
    .apt_install(
        "cmake",
        "pkg-config",
        "build-essential",
        "curl",
        "git",
        "libssl-dev",
        "ffmpeg",
    )
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable")
    .run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc")
    .env({
        "PATH": "/root/.cargo/bin:$PATH",
        "FASTVIDEO_KERNEL_CACHE_ROOT": KERNEL_CACHE_VOLUME_PATH,
        **({"UV_TORCH_BACKEND": uv_torch_backend_override} if uv_torch_backend_override else {}),
    })
)

COMMON_KWARGS = dict(
    gpu="L40S:1",
    image=image,
    timeout=5400,
    volumes={KERNEL_CACHE_VOLUME_PATH: kernel_cache_vol},
)


def _run_local_git_command(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _normalize_git_repo_url(git_repo: str) -> str:
    if git_repo.startswith("git@github.com:"):
        return "https://github.com/" + git_repo[len("git@github.com:"):]
    if git_repo.startswith("ssh://git@github.com/"):
        return "https://github.com/" + git_repo[len("ssh://git@github.com/"):]
    return git_repo


def _resolve_git_repo(git_repo: str) -> str:
    if git_repo.strip():
        return _normalize_git_repo_url(git_repo.strip())
    env_repo = os.environ.get("BUILDKITE_REPO", "").strip()
    if env_repo:
        return _normalize_git_repo_url(env_repo)
    discovered_repo = _run_local_git_command(["config", "--get", "remote.origin.url"])
    if discovered_repo:
        return _normalize_git_repo_url(discovered_repo)
    raise RuntimeError("Could not resolve git repo URL. Pass --git-repo or set BUILDKITE_REPO.")


def _resolve_git_commit(git_commit: str) -> str:
    if git_commit.strip():
        return git_commit.strip()
    env_commit = os.environ.get("BUILDKITE_COMMIT", "").strip()
    if env_commit:
        return env_commit
    discovered_commit = _run_local_git_command(["rev-parse", "HEAD"])
    if discovered_commit:
        return discovered_commit
    raise RuntimeError("Could not resolve git commit. Pass --git-commit or set BUILDKITE_COMMIT.")


def _run(args: list[str], cwd: str | None = None) -> str:
    print("$ " + " ".join(shlex.quote(arg) for arg in args), flush=True)
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    return result.stdout.strip()


def _run_shell_capture(command: str, cwd: str) -> str:
    print(f"$ {command}", flush=True)
    result = subprocess.run(
        ["/bin/bash", "-lc", command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")
    return result.stdout


def _clone_checkout(git_repo: str, git_commit: str) -> str:
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    _run(["git", "-c", "http.version=HTTP/1.1", "clone", git_repo, REPO_DIR], cwd="/")
    _run(["git", "checkout", git_commit], cwd=REPO_DIR)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO_DIR)
    return _run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)


def _cache_install_output(cache_root: str) -> str:
    # This smoke exercises the Modal volume cache specifically, so do not let
    # a matching Docker-prebuilt wheel satisfy the miss/hit assertions.
    disabled_prebuilt_info_path = f"{cache_root}/.docker-prebuilt-disabled.json"
    return _run_shell_capture(
        "source $HOME/.local/bin/env && "
        "source /opt/venv/bin/activate && "
        "python fastvideo/tests/modal/kernel_build_cache.py install "
        f"--cache-root {shlex.quote(cache_root)} "
        f"--prebuilt-info-path {shlex.quote(disabled_prebuilt_info_path)}",
        cwd=REPO_DIR,
    )


def _assert_in_output(output: str, needle: str) -> None:
    if needle not in output:
        raise RuntimeError(f"Expected output to contain {needle!r}")


def _assert_not_in_output(output: str, needle: str) -> None:
    if needle in output:
        raise RuntimeError(f"Expected output not to contain {needle!r}")


@app.function(**COMMON_KWARGS)
def build_and_store_cache(git_repo: str, git_commit: str, cache_root: str) -> dict[str, Any]:
    checked_out_commit = _clone_checkout(git_repo, git_commit)
    output = _cache_install_output(cache_root)
    _assert_in_output(output, "cache miss:")
    _assert_in_output(output, "./build.sh --wheel-dir")
    _assert_in_output(output, "stored wheel cache entry:")
    print("Committing kernel cache volume after miss/store", flush=True)
    kernel_cache_vol.commit()
    return {
        "checked_out_commit": checked_out_commit,
        "cache_root": cache_root,
        "cache_miss": True,
        "stored": True,
    }


@app.function(**COMMON_KWARGS)
def verify_cache_hit(git_repo: str, git_commit: str, cache_root: str) -> dict[str, Any]:
    checked_out_commit = _clone_checkout(git_repo, git_commit)
    output = _cache_install_output(cache_root)
    _assert_in_output(output, "cache hit:")
    _assert_not_in_output(output, "cache miss:")
    _assert_not_in_output(output, "./build.sh --wheel-dir")
    return {
        "checked_out_commit": checked_out_commit,
        "cache_root": cache_root,
        "cache_hit": True,
    }


@app.function(image=image, timeout=300, volumes={KERNEL_CACHE_VOLUME_PATH: kernel_cache_vol})
def cleanup_cache(cache_root: str) -> None:
    shutil.rmtree(cache_root, ignore_errors=True)
    print(f"Removed smoke cache root {cache_root}", flush=True)
    kernel_cache_vol.commit()


@app.local_entrypoint()
def main(
    git_repo: str = "",
    git_commit: str = "",
    cache_namespace: str = "",
    cleanup: bool = True,
):
    resolved_git_repo = _resolve_git_repo(git_repo)
    resolved_git_commit = _resolve_git_commit(git_commit)
    namespace = cache_namespace.strip() or f"smoke-{resolved_git_commit[:12]}-{int(time.time())}"
    cache_root = f"{KERNEL_CACHE_VOLUME_PATH}/{namespace}"

    print("Launching cross-container kernel cache smoke")
    print(f"Repo: {resolved_git_repo}")
    print(f"Commit: {resolved_git_commit}")
    print(f"Cache root: {cache_root}")

    miss_result = build_and_store_cache.remote(resolved_git_repo, resolved_git_commit, cache_root)
    hit_result = verify_cache_hit.remote(resolved_git_repo, resolved_git_commit, cache_root)
    if cleanup:
        cleanup_cache.remote(cache_root)

    print("Kernel cache smoke passed")
    print({"miss": miss_result, "hit": hit_result, "cleanup": cleanup})
