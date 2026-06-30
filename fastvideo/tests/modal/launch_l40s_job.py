"""Launch an arbitrary FastVideo command on Modal GPUs.

Examples:
    python -m modal run fastvideo/tests/modal/launch_l40s_job.py --command "nvidia-smi" --install-extra none

    python -m modal run fastvideo/tests/modal/launch_l40s_job.py \
        --num-gpus 2 \
        --install-extra test \
        --command "pytest fastvideo/tests/vaes -vs"

    python -m modal run fastvideo/tests/modal/launch_l40s_job.py \
        --gpu-type H100 \
        --num-gpus 1 \
        --install-extra none \
        --command "nvidia-smi"

Use ``--no-wait`` with ``modal run --detach`` when the job should keep running
after the local Modal client exits.
"""

import os
import base64
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import modal

app = modal.App("fastvideo-gpu-job")

REPO_DIR = "/FastVideo"
MODEL_VOLUME_NAME = os.environ.get("FASTVIDEO_MODAL_VOLUME", "hf-model-weights")
IMAGE_VERSION = os.environ.get("IMAGE_VERSION", "latest")
IMAGE_TAG = os.environ.get(
    "FASTVIDEO_MODAL_IMAGE",
    f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{IMAGE_VERSION}",
)
SECRET_ENV_KEYS = (
    "HF_API_KEY",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
    "WANDB_MODE",
)

print(f"Using image: {IMAGE_TAG}")
print(f"Using Modal volume: {MODEL_VOLUME_NAME}")

model_vol = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
local_secrets = modal.Secret.from_dict({
    key: os.environ[key]
    for key in SECRET_ENV_KEYS
    if os.environ.get(key)
})

# Mutable tags inherit the registry image's baked backend, including custom
# FASTVIDEO_MODAL_IMAGE overrides. Explicit CUDA tags also work with older
# images that predate the baked setting, and a caller override always wins.
uv_torch_backend_override = os.environ.get("UV_TORCH_BACKEND")
if not uv_torch_backend_override:
    if "cuda13" in IMAGE_TAG.lower():
        uv_torch_backend_override = "cu130"
    elif "cuda12.6" in IMAGE_TAG.lower():
        uv_torch_backend_override = "cu126"

image = (
    modal.Image.from_registry(IMAGE_TAG, add_python="3.12")
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
        "HF_HOME": "/root/data/.cache",
        "TOKENIZERS_PARALLELISM": "false",
        **({"UV_TORCH_BACKEND": uv_torch_backend_override} if uv_torch_backend_override else {}),
        "FASTVIDEO_ATTENTION_BACKEND": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN"),
    })
)

COMMON_FUNCTION_KWARGS = dict(
    image=image,
    timeout=86400,
    secrets=[local_secrets],
    volumes={"/root/data": model_vol},
)


def _run_local_git_command(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_local_git_command_allow_diff(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _split_patch_paths(patch_paths: str) -> list[str]:
    return [path.strip() for path in patch_paths.split(",") if path.strip()]


def _build_local_patch(patch_paths: str) -> str:
    paths = _split_patch_paths(patch_paths)
    diff_args = ["diff", "--binary"]
    if paths:
        diff_args.extend(["--", *paths])
    patch_parts = [_run_local_git_command_allow_diff(diff_args)]

    untracked_args = ["ls-files", "--others", "--exclude-standard"]
    if paths:
        untracked_args.extend(["--", *paths])
    untracked = _run_local_git_command(untracked_args).splitlines()
    for path in untracked:
        if not os.path.isfile(path):
            continue
        patch_parts.append(_run_local_git_command_allow_diff(["diff", "--binary", "--no-index", "/dev/null", path]))

    patch = "\n".join(part for part in patch_parts if part.strip())
    if not patch.strip():
        raise RuntimeError("Requested --apply-local-patch but no local diff was found.")
    return patch


def _apply_local_patch(patch_b64: str) -> None:
    if not patch_b64:
        return
    patch = base64.b64decode(patch_b64.encode("ascii"))
    print("Applying local workspace patch", flush=True)
    result = subprocess.run(
        ["git", "apply", "--binary", "--whitespace=nowarn", "-"],
        cwd=REPO_DIR,
        input=patch,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        check=False,
    )
    if result.stdout:
        print(result.stdout.decode("utf-8", errors="replace"), end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to apply local patch with exit code {result.returncode}")


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


def _resolve_pull_request(pr_number: str) -> str:
    if pr_number.strip():
        return pr_number.strip()
    env_pr = os.environ.get("BUILDKITE_PULL_REQUEST", "").strip()
    if env_pr:
        return env_pr
    return "false"


def _run(args: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> str:
    print("$ " + " ".join(shlex.quote(arg) for arg in args), flush=True)
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    return result.stdout.strip()


def _run_shell(command: str, cwd: str, env: dict[str, str]) -> None:
    print(f"$ {command}", flush=True)
    result = subprocess.run(
        ["/bin/bash", "-lc", command],
        cwd=cwd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")


def _parse_env_vars(env_vars: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in env_vars.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"Invalid env var override {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _activate_remote_python_env(env: dict[str, str]) -> dict[str, str]:
    venv_bin = "/opt/venv/bin"
    if os.path.isdir(venv_bin):
        env["VIRTUAL_ENV"] = "/opt/venv"
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    return env


def _clone_checkout(git_repo: str, git_commit: str, pr_number: str) -> str:
    last_clone_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, 4):
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        try:
            _run(
                [
                    "git",
                    "-c",
                    "http.version=HTTP/1.1",
                    "clone",
                    git_repo,
                    REPO_DIR,
                ],
                cwd="/",
            )
            break
        except subprocess.CalledProcessError as error:
            last_clone_error = error
            if attempt == 3:
                raise
            sleep_seconds = 5 * attempt
            print(
                f"git clone failed on attempt {attempt}; retrying in {sleep_seconds}s",
                flush=True,
            )
            time.sleep(sleep_seconds)
    if last_clone_error is not None and not os.path.isdir(REPO_DIR):
        raise last_clone_error
    if pr_number and pr_number != "false":
        _run(["git", "fetch", "--prune", "origin", f"refs/pull/{pr_number}/head"], cwd=REPO_DIR)
        _run(["git", "checkout", "FETCH_HEAD"], cwd=REPO_DIR)
    else:
        _run(["git", "checkout", git_commit], cwd=REPO_DIR)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO_DIR)
    return _run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)


def _install_fastvideo(install_extra: str, env: dict[str, str]) -> None:
    install_extra = install_extra.strip()
    if install_extra.lower() in {"", "none", "skip", "false"}:
        return
    package = "." if install_extra == "." else f".[{install_extra}]"
    _run_shell(
        "source $HOME/.local/bin/env 2>/dev/null || true; "
        "source /opt/venv/bin/activate 2>/dev/null || true; "
        f"uv pip install -e {shlex.quote(package)}",
        cwd=REPO_DIR,
        env=env,
    )


def _build_kernel(env: dict[str, str]) -> None:
    _run_shell(
        "source $HOME/.local/bin/env 2>/dev/null || true; "
        "source /opt/venv/bin/activate 2>/dev/null || true; "
        "./build.sh",
        cwd=os.path.join(REPO_DIR, "fastvideo-kernel"),
        env=env,
    )


def _run_gpu_job(
    command: str,
    git_repo: str,
    git_commit: str,
    pr_number: str,
    install_extra: str,
    build_kernel: bool,
    env_vars: str,
    local_patch_b64: str,
    commit_volume: bool,
) -> dict[str, Any]:
    remote_env = _activate_remote_python_env(os.environ.copy())
    remote_env.update({
        "HF_HOME": "/root/data/.cache",
        "TOKENIZERS_PARALLELISM": "false",
        "FASTVIDEO_ATTENTION_BACKEND": remote_env.get("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN"),
    })
    remote_env.update(_parse_env_vars(env_vars))

    print(f"Cloning repository: {git_repo}")
    print(f"Target commit: {git_commit}")
    if pr_number and pr_number != "false":
        print(f"Using PR ref: {pr_number}")
    checked_out_commit = _clone_checkout(git_repo, git_commit, pr_number)
    print(f"Checked out commit: {checked_out_commit}")
    _apply_local_patch(local_patch_b64)

    _install_fastvideo(install_extra, remote_env)
    if build_kernel:
        _build_kernel(remote_env)

    try:
        _run_shell(command, cwd=REPO_DIR, env=remote_env)
    finally:
        if commit_volume:
            print("Committing Modal volume", flush=True)
            model_vol.commit()
    return {
        "command": command,
        "git_repo": git_repo,
        "git_commit": checked_out_commit,
        "install_extra": install_extra,
        "build_kernel": build_kernel,
        "local_patch_applied": bool(local_patch_b64),
        "commit_volume": commit_volume,
    }


@app.function(gpu="L40S:1", **COMMON_FUNCTION_KWARGS)
def run_l40s_1(
    command: str,
    git_repo: str,
    git_commit: str,
    pr_number: str,
    install_extra: str,
    build_kernel: bool,
    env_vars: str,
    local_patch_b64: str,
    commit_volume: bool,
):
    return _run_gpu_job(command, git_repo, git_commit, pr_number, install_extra, build_kernel, env_vars,
                        local_patch_b64, commit_volume)


@app.function(gpu="L40S:2", **COMMON_FUNCTION_KWARGS)
def run_l40s_2(
    command: str,
    git_repo: str,
    git_commit: str,
    pr_number: str,
    install_extra: str,
    build_kernel: bool,
    env_vars: str,
    local_patch_b64: str,
    commit_volume: bool,
):
    return _run_gpu_job(command, git_repo, git_commit, pr_number, install_extra, build_kernel, env_vars,
                        local_patch_b64, commit_volume)


@app.function(gpu="L40S:4", **COMMON_FUNCTION_KWARGS)
def run_l40s_4(
    command: str,
    git_repo: str,
    git_commit: str,
    pr_number: str,
    install_extra: str,
    build_kernel: bool,
    env_vars: str,
    local_patch_b64: str,
    commit_volume: bool,
):
    return _run_gpu_job(command, git_repo, git_commit, pr_number, install_extra, build_kernel, env_vars,
                        local_patch_b64, commit_volume)


@app.function(gpu="L40S:8", **COMMON_FUNCTION_KWARGS)
def run_l40s_8(
    command: str,
    git_repo: str,
    git_commit: str,
    pr_number: str,
    install_extra: str,
    build_kernel: bool,
    env_vars: str,
    local_patch_b64: str,
    commit_volume: bool,
):
    return _run_gpu_job(command, git_repo, git_commit, pr_number, install_extra, build_kernel, env_vars,
                        local_patch_b64, commit_volume)


@app.function(gpu="H100:1", **COMMON_FUNCTION_KWARGS)
def run_h100_1(
    command: str,
    git_repo: str,
    git_commit: str,
    pr_number: str,
    install_extra: str,
    build_kernel: bool,
    env_vars: str,
    local_patch_b64: str,
    commit_volume: bool,
):
    return _run_gpu_job(command, git_repo, git_commit, pr_number, install_extra, build_kernel, env_vars,
                        local_patch_b64, commit_volume)


@app.function(gpu="H100:2", **COMMON_FUNCTION_KWARGS)
def run_h100_2(
    command: str,
    git_repo: str,
    git_commit: str,
    pr_number: str,
    install_extra: str,
    build_kernel: bool,
    env_vars: str,
    local_patch_b64: str,
    commit_volume: bool,
):
    return _run_gpu_job(command, git_repo, git_commit, pr_number, install_extra, build_kernel, env_vars,
                        local_patch_b64, commit_volume)


def _select_runner(gpu_type: str, num_gpus: int) -> Callable[..., Any]:
    normalized_gpu_type = gpu_type.upper()
    runners = {
        ("L40S", 1): run_l40s_1,
        ("L40S", 2): run_l40s_2,
        ("L40S", 4): run_l40s_4,
        ("L40S", 8): run_l40s_8,
        ("H100", 1): run_h100_1,
        ("H100", 2): run_h100_2,
    }
    try:
        return runners[(normalized_gpu_type, num_gpus)]
    except KeyError as error:
        supported = ", ".join(f"{gpu}:{count}" for gpu, count in sorted(runners))
        raise RuntimeError(f"Unsupported GPU request {gpu_type}:{num_gpus}. Supported requests: {supported}.") from error


@app.local_entrypoint()
def main(
    command: str = "nvidia-smi",
    gpu_type: str = "L40S",
    num_gpus: int = 1,
    git_repo: str = "",
    git_commit: str = "",
    pr_number: str = "",
    install_extra: str = "dev",
    build_kernel: bool = False,
    env_vars: str = "",
    apply_local_patch: bool = False,
    patch_paths: str = "",
    wait: bool = True,
    commit_volume: bool = False,
):
    normalized_gpu_type = gpu_type.upper()
    resolved_git_repo = _resolve_git_repo(git_repo)
    resolved_git_commit = _resolve_git_commit(git_commit)
    resolved_pr_number = _resolve_pull_request(pr_number)
    runner = _select_runner(normalized_gpu_type, num_gpus)

    print(f"Launching {normalized_gpu_type}:{num_gpus} job")
    print(f"Command: {command}")
    print(f"Repo: {resolved_git_repo}")
    print(f"Commit: {resolved_git_commit}")
    if resolved_pr_number and resolved_pr_number != "false":
        print(f"PR ref: {resolved_pr_number}")
    local_patch_b64 = ""
    if apply_local_patch:
        patch = _build_local_patch(patch_paths)
        local_patch_b64 = base64.b64encode(patch.encode("utf-8")).decode("ascii")
        print(f"Local patch payload: {len(patch)} bytes")

    kwargs = dict(
        command=command,
        git_repo=resolved_git_repo,
        git_commit=resolved_git_commit,
        pr_number=resolved_pr_number,
        install_extra=install_extra,
        build_kernel=build_kernel,
        env_vars=env_vars,
        local_patch_b64=local_patch_b64,
        commit_volume=commit_volume,
    )
    if wait:
        result = runner.remote(**kwargs)
        print(f"Completed {normalized_gpu_type} job: {result}")
        return

    function_call = runner.spawn(**kwargs)
    print(f"Spawned Modal FunctionCall: {function_call.object_id}")
    print("Poll later with:")
    print(f"  python -c \"import modal; print(modal.FunctionCall.from_id('{function_call.object_id}').get())\"")
