import ast
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import modal

app = modal.App()

model_vol = modal.Volume.from_name("hf-model-weights")
image_version = os.getenv("IMAGE_VERSION", "latest")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
print(f"Using image: {image_tag}")

image = (
    modal.Image.from_registry(image_tag, add_python="3.12")
    .apt_install(
        "cmake",
        "pkg-config",
        "build-essential",
        "curl",
        "libssl-dev",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable")
    .run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc")
    .env(
        {
            "PATH": "/root/.cargo/bin:$PATH",
            "BUILDKITE_REPO": os.environ.get("BUILDKITE_REPO", ""),
            "BUILDKITE_COMMIT": os.environ.get("BUILDKITE_COMMIT", ""),
            "BUILDKITE_PULL_REQUEST": os.environ.get("BUILDKITE_PULL_REQUEST", ""),
            "IMAGE_VERSION": os.environ.get("IMAGE_VERSION", ""),
        }
    )
)

SSIM_NUM_GPUS = 4
SSIM_TERMINATE_TIMEOUT_S = 30
HF_TOKEN_ENV_KEYS = ("HF_API_KEY", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")
RAW_GENERATED_VOLUME_ROOT = "ssim_generated_videos"
DEFAULT_OUTPUT_QUALITY_TIER = "default"
FULL_OUTPUT_QUALITY_TIER = "full_quality"
MODAL_DEVICE_REFERENCE_FOLDER = "L40S_reference_videos"
SSIM_COMMON_KWARGS = dict(
    image=image,
    timeout=5400,
    volumes={"/root/data": model_vol},
)


@dataclass(frozen=True)
class SSIMTask:
    task_id: int
    test_file: str
    required_gpus: int
    model_id: str | None = None

    @property
    def test_name(self) -> str:
        test_file_name = os.path.basename(self.test_file)
        if self.model_id is None:
            return test_file_name
        return f"{test_file_name}::{self.model_id}"

    @property
    def sort_key(self) -> tuple[str, str]:
        return (os.path.basename(self.test_file), self.model_id or "")


@dataclass
class _RunningTask:
    task: SSIMTask
    process: Any
    gpu_ids: list[str]
    log_path: str
    log_handle: Any


@dataclass
class _TaskResult:
    task: SSIMTask
    status: str
    returncode: int
    gpu_ids: list[str]
    log_path: str | None = None


@dataclass
class _TaskSummary:
    test_name: str
    required_gpus: int
    status: str
    returncode: int
    log_content: str | None = None


@dataclass
class _PartitionResult:
    partition_index: int
    task_summaries: list[_TaskSummary]
    exit_code: int


def _split_csv_values(csv_values: str) -> set[str]:
    return {value.strip() for value in csv_values.split(",") if value.strip()}


def _run_git_command(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        raise RuntimeError(f"Failed to run git {' '.join(args)}: {stderr or error}") from error
    return result.stdout.strip()


def _normalize_git_repo_url(git_repo: str) -> str:
    if git_repo.startswith("git@github.com:"):
        return "https://github.com/" + git_repo[len("git@github.com:") :]
    if git_repo.startswith("ssh://git@github.com/"):
        return "https://github.com/" + git_repo[len("ssh://git@github.com/") :]
    return git_repo


def _resolve_git_repo(git_repo: str) -> str:
    if git_repo.strip():
        return _normalize_git_repo_url(git_repo.strip())

    env_repo = os.environ.get("BUILDKITE_REPO", "").strip()
    if env_repo:
        return _normalize_git_repo_url(env_repo)

    discovered_repo = _run_git_command(["config", "--get", "remote.origin.url"])
    if discovered_repo:
        return _normalize_git_repo_url(discovered_repo)

    raise RuntimeError("Could not resolve git repo URL. Pass --git-repo or set BUILDKITE_REPO.")


def _resolve_git_commit(git_commit: str) -> str:
    if git_commit.strip():
        return git_commit.strip()

    env_commit = os.environ.get("BUILDKITE_COMMIT", "").strip()
    if env_commit:
        return env_commit

    discovered_commit = _run_git_command(["rev-parse", "HEAD"])
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


def _resolve_hf_api_key(
    hf_api_key: str,
) -> str:
    if hf_api_key.strip():
        return hf_api_key.strip()

    for key in HF_TOKEN_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value

    raise RuntimeError(
        "Hugging Face token is required. Set HF_API_KEY, HUGGINGFACE_HUB_TOKEN, or HF_TOKEN; or pass --hf-api-key."
    )


def _sanitize_path_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-")


def _resolve_generated_volume_subdir(
    requested_subdir: str,
    git_commit: str,
) -> str:
    if requested_subdir.strip():
        normalized = requested_subdir.strip().strip("/")
        if not normalized:
            raise RuntimeError("generated_volume_subdir must not be empty.")
        return normalized

    short_commit = _sanitize_path_fragment(git_commit[:12]) or "unknown"
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{short_commit}"


def _resolve_output_quality_tier(ssim_full_quality: bool) -> str:
    if ssim_full_quality:
        return FULL_OUTPUT_QUALITY_TIER
    return DEFAULT_OUTPUT_QUALITY_TIER


def _build_generated_volume_relative_path(
    *,
    generated_volume_subdir: str,
    quality_tier: str,
) -> str:
    return os.path.join(
        RAW_GENERATED_VOLUME_ROOT,
        quality_tier,
        generated_volume_subdir,
        "generated_videos",
    )


def _build_local_generated_download_dir(quality_tier: str) -> str:
    return os.path.join(".", "generated_videos_modal", quality_tier)


def _print_local_reference_copy_command(quality_tier: str) -> None:
    generated_dir = os.path.join(
        _build_local_generated_download_dir(quality_tier),
        MODAL_DEVICE_REFERENCE_FOLDER,
    )
    print(
        "To update local references from the downloaded Modal outputs, run:\n"
        "  python fastvideo/tests/ssim/reference_videos_cli.py copy-local "
        f"--quality-tier {quality_tier} "
        f"--generated-dir {generated_dir} "
        f"--device-folder {MODAL_DEVICE_REFERENCE_FOLDER}"
    )


def _count_video_files(root: str) -> int:
    count = 0
    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")):
                count += 1
    return count


def _sync_generated_videos_to_volume(
    repo_root: str,
    generated_volume_subdir: str,
    quality_tier: str,
) -> str | None:
    generated_root = os.path.join(
        repo_root,
        "fastvideo",
        "tests",
        "ssim",
        "generated_videos",
        quality_tier,
    )
    if not os.path.isdir(generated_root):
        print(
            f"No generated_videos directory found for quality tier {quality_tier}; skipping raw generated video export."
        )
        return None

    relative_dst = _build_generated_volume_relative_path(
        generated_volume_subdir=generated_volume_subdir,
        quality_tier=quality_tier,
    )
    absolute_dst = os.path.join("/root/data", relative_dst)
    if os.path.exists(absolute_dst):
        shutil.rmtree(absolute_dst)
    os.makedirs(os.path.dirname(absolute_dst), exist_ok=True)
    shutil.copytree(generated_root, absolute_dst)
    model_vol.commit()

    num_videos = _count_video_files(absolute_dst)
    print(f"Raw generated videos exported to Modal volume path: {relative_dst} ({num_videos} video files).")
    print(
        "Download command:\n"
        f"  modal volume get hf-model-weights {relative_dst} "
        f"{_build_local_generated_download_dir(quality_tier)}"
    )
    _print_local_reference_copy_command(quality_tier)
    return relative_dst


def _extract_required_gpus(filepath: str) -> int:
    """Read REQUIRED_GPUS from a test file. Defaults to 1."""
    with open(filepath, encoding="utf-8") as file:
        for line in file:
            match = re.match(r"^REQUIRED_GPUS\s*=\s*(\d+)", line)
            if match:
                return int(match.group(1))
    return 1


def _extract_model_ids(filepath: str) -> list[str]:
    """Extract model ids from *_MODEL_TO_PARAMS dictionaries."""
    with open(filepath, encoding="utf-8") as file:
        module_ast = ast.parse(file.read(), filename=filepath)

    model_ids = []
    for node in module_ast.body:
        target_names = []
        value_node = None

        if isinstance(node, ast.Assign):
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target_names = [node.target.id]
            value_node = node.value

        if not target_names or value_node is None:
            continue
        if not any(name.endswith("MODEL_TO_PARAMS") for name in target_names):
            continue
        if not isinstance(value_node, ast.Dict):
            continue

        for key_node in value_node.keys:
            if isinstance(key_node, ast.Constant):
                if isinstance(key_node.value, str):
                    model_ids.append(key_node.value)

    unique_model_ids = []
    seen_model_ids = set()
    for model_id in model_ids:
        if model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)
        unique_model_ids.append(model_id)
    return unique_model_ids


def _discover_ssim_tasks(
    ssim_dir: str,
    *,
    selected_test_files: set[str] | None = None,
    selected_model_ids: set[str] | None = None,
) -> list[SSIMTask]:
    tasks = []
    task_id = 0
    selected_test_files = selected_test_files or set()
    selected_test_file_names = {os.path.basename(test_file) for test_file in selected_test_files}
    selected_model_ids = selected_model_ids or set()

    test_files = sorted(glob.glob(os.path.join(ssim_dir, "test_*.py")))
    matched_test_files = set()
    matched_model_ids = set()
    for filepath in test_files:
        file_name = os.path.basename(filepath)
        rel_path = f"./fastvideo/tests/ssim/{file_name}"
        if selected_test_files and (
            filepath not in selected_test_files
            and rel_path not in selected_test_files
            and file_name not in selected_test_file_names
        ):
            continue

        matched_test_files.add(file_name)
        required_gpus = _extract_required_gpus(filepath)
        if required_gpus < 1 or required_gpus > SSIM_NUM_GPUS:
            raise ValueError(f"{filepath} requires {required_gpus} GPUs, but scheduler supports up to {SSIM_NUM_GPUS}.")
        model_ids = _extract_model_ids(filepath)
        if model_ids:
            for model_id in model_ids:
                if selected_model_ids and model_id not in selected_model_ids:
                    continue
                matched_model_ids.add(model_id)
                tasks.append(
                    SSIMTask(
                        task_id=task_id,
                        test_file=rel_path,
                        required_gpus=required_gpus,
                        model_id=model_id,
                    )
                )
                task_id += 1
        else:
            if selected_model_ids:
                continue
            tasks.append(
                SSIMTask(
                    task_id=task_id,
                    test_file=rel_path,
                    required_gpus=required_gpus,
                )
            )
            task_id += 1

    if selected_test_files:
        unmatched_files = sorted(selected_test_file_names - matched_test_files)
        if unmatched_files:
            raise RuntimeError("Requested SSIM test file(s) not found: " + ", ".join(unmatched_files))
    if selected_model_ids:
        unmatched_model_ids = sorted(selected_model_ids - matched_model_ids)
        if unmatched_model_ids:
            raise RuntimeError("Requested SSIM model_id(s) not found: " + ", ".join(unmatched_model_ids))
    return sorted(tasks, key=lambda task: task.sort_key)


def _partition_tasks(
    tasks: list[SSIMTask],
    partition_index: int,
    num_partitions: int = 2,
) -> list[SSIMTask]:
    """Split tasks into N groups via round-robin on sorted order."""
    return tasks[partition_index::num_partitions]


def _build_checkout_command(git_commit: str, pr_number: str | None) -> str:
    import shlex

    if pr_number and pr_number != "false":
        try:
            pr_id = int(pr_number)
        except ValueError as error:
            raise RuntimeError(f"Invalid BUILDKITE_PULL_REQUEST value: {pr_number}") from error
        return f"git fetch --prune origin refs/pull/{pr_id}/head && git checkout FETCH_HEAD"
    return f"git checkout {shlex.quote(git_commit)}"


def _prepare_ssim_workspace(
    *,
    git_repo: str,
    git_commit: str,
    pr_number: str,
    hf_api_key: str,
    selected_test_files: set[str] | None = None,
    selected_model_ids: set[str] | None = None,
) -> tuple[str, list[SSIMTask]]:
    import shlex

    if not hf_api_key.strip():
        raise RuntimeError("HF API key is required to prepare SSIM workspace.")

    checkout_command = _build_checkout_command(git_commit, pr_number)
    repo_root = "/FastVideo"

    command = f"""
    set -euo pipefail
    source $HOME/.local/bin/env
    source /opt/venv/bin/activate
    git_retry() {{
      local attempt
      for attempt in 1 2 3; do
        if "$@"; then return 0; fi
        echo "git command failed (attempt $attempt/3), retrying in 5s..."
        sleep 5
      done
      "$@"
    }}
    if [ -d {shlex.quote(repo_root)}/.git ]; then
      cd {shlex.quote(repo_root)}
      git remote set-url origin {shlex.quote(git_repo)} || true
      git_retry git fetch --prune origin
    else
      git_retry git clone {shlex.quote(git_repo)} {shlex.quote(repo_root)}
      cd {shlex.quote(repo_root)}
    fi
    {checkout_command}
    rm -rf fastvideo/tests/ssim/reference_videos
    git_retry git submodule update --init --recursive
    uv pip install -e ".[test]"
    cd fastvideo-kernel
    ./build.sh
    cd ..
    uv pip install git+https://github.com/microsoft/MoGe.git
    # Stable Audio Open 1.0 inference deps (optional in basic install,
    # required by `StableAudioDenoisingStage`; consumed by
    # `test_stable_audio_similarity.py`).
    uv pip install k_diffusion einops_exts alias_free_torch torchsde
    export HF_HOME='/root/data/.cache'
    hf auth login --token "$HF_API_KEY"
    """
    result = subprocess.run(
        ["/bin/bash", "-lc", command],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HF_API_KEY": hf_api_key,
        },
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Workspace setup failed with exit code {result.returncode}")

    ssim_dir = os.path.join(repo_root, "fastvideo", "tests", "ssim")
    tasks = _discover_ssim_tasks(
        ssim_dir,
        selected_test_files=selected_test_files,
        selected_model_ids=selected_model_ids,
    )
    if not tasks:
        raise RuntimeError("No SSIM test files found.")
    return repo_root, tasks


def _spawn_ssim_task(
    task: SSIMTask,
    repo_root: str,
    assigned_gpu_ids: list[str],
    log_dir: str,
    task_index: int,
    pytest_extra_args: list[str],
) -> _RunningTask:
    import shlex

    safe_test_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.test_name)
    log_path = os.path.join(log_dir, f"{task_index:03d}_{safe_test_name}.log")
    pytest_command = shlex.join(["pytest", task.test_file, "-vs", *pytest_extra_args])
    command = f"set -euo pipefail && source $HOME/.local/bin/env && source /opt/venv/bin/activate && {pytest_command}"
    env = os.environ.copy()
    env["HF_HOME"] = "/root/data/.cache"
    # MultiprocExecutor returns CUDA tensors through mp pipes (CUDA IPC).
    # On kernels without pidfd_open support, PyTorch fails when
    # expandable_segments=True. Force False for CI compatibility.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned_gpu_ids)
    if task.model_id is None:
        env.pop("FASTVIDEO_SSIM_MODEL_ID", None)
    else:
        env["FASTVIDEO_SSIM_MODEL_ID"] = task.model_id

    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        cwd=repo_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return _RunningTask(
        task=task,
        process=process,
        gpu_ids=assigned_gpu_ids,
        log_path=log_path,
        log_handle=log_handle,
    )


def _finalize_running_task(
    running_task: _RunningTask,
    returncode: int,
    status: str,
    results: dict[int, _TaskResult],
    available_gpu_ids: list[str],
    gpu_order: dict[str, int],
) -> None:
    running_task.log_handle.close()
    available_gpu_ids.extend(running_task.gpu_ids)
    available_gpu_ids.sort(key=lambda gpu_id: gpu_order.get(gpu_id, len(gpu_order)))
    results[running_task.task.task_id] = _TaskResult(
        task=running_task.task,
        status=status,
        returncode=returncode,
        gpu_ids=running_task.gpu_ids,
        log_path=running_task.log_path,
    )


def _terminate_running_tasks(
    running_tasks: list[_RunningTask],
    results: dict[int, _TaskResult],
    available_gpu_ids: list[str],
    gpu_order: dict[str, int],
) -> None:
    import signal
    import time

    for running_task in running_tasks:
        if running_task.process.poll() is None:
            try:
                os.killpg(running_task.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    deadline = time.time() + SSIM_TERMINATE_TIMEOUT_S
    while time.time() < deadline:
        if all(task.process.poll() is not None for task in running_tasks):
            break
        time.sleep(1)

    for running_task in running_tasks:
        if running_task.process.poll() is None:
            try:
                os.killpg(running_task.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    for running_task in list(running_tasks):
        returncode = running_task.process.wait()
        status = "terminated"
        if returncode == 0:
            status = "passed"
        _finalize_running_task(
            running_task=running_task,
            returncode=returncode,
            status=status,
            results=results,
            available_gpu_ids=available_gpu_ids,
            gpu_order=gpu_order,
        )
        running_tasks.remove(running_task)


def _get_visible_gpu_ids() -> list[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        return [str(index) for index in range(SSIM_NUM_GPUS)]

    gpu_ids = []
    seen_gpu_ids: set[str] = set()
    for gpu_id in cuda_visible_devices.split(","):
        cleaned_gpu_id = gpu_id.strip()
        if not cleaned_gpu_id or cleaned_gpu_id in seen_gpu_ids:
            continue
        seen_gpu_ids.add(cleaned_gpu_id)
        gpu_ids.append(cleaned_gpu_id)
    if not gpu_ids:
        return [str(index) for index in range(SSIM_NUM_GPUS)]
    return gpu_ids


def _build_pytest_extra_args(
    *,
    ssim_full_quality: bool,
    ssim_reference_repo: str,
    skip_ssim_reference_download: bool,
    pytest_k: str,
) -> list[str]:
    args = []
    if ssim_full_quality:
        args.append("--ssim-full-quality")
    if ssim_reference_repo.strip():
        args.extend(["--ssim-reference-repo", ssim_reference_repo.strip()])
    if skip_ssim_reference_download:
        args.append("--skip-ssim-reference-download")
    if pytest_k.strip():
        args.extend(["-k", pytest_k.strip()])
    return args


def _schedule_ssim_tasks(
    repo_root: str,
    tasks: list[SSIMTask],
    pytest_extra_args: list[str],
    fail_fast: bool = True,
) -> dict[int, _TaskResult]:
    import tempfile
    import time

    pending_tasks = list(tasks)
    running_tasks = []
    available_gpu_ids = _get_visible_gpu_ids()
    gpu_order = {gpu_id: index for index, gpu_id in enumerate(available_gpu_ids)}
    max_required_gpus = max((task.required_gpus for task in pending_tasks), default=0)
    if max_required_gpus > len(available_gpu_ids):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise RuntimeError(
            "SSIM task requires "
            f"{max_required_gpus} GPUs but only "
            f"{len(available_gpu_ids)} are visible via "
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}."
        )
    results = {}
    fail_fast_triggered = False
    log_dir = tempfile.mkdtemp(prefix="fastvideo-ssim-logs-")

    while pending_tasks or running_tasks:
        while not fail_fast_triggered:
            next_task_index = None
            for index, task in enumerate(pending_tasks):
                if task.required_gpus <= len(available_gpu_ids):
                    next_task_index = index
                    break
            if next_task_index is None:
                break

            task = pending_tasks.pop(next_task_index)
            assigned_gpu_ids = available_gpu_ids[: task.required_gpus]
            del available_gpu_ids[: task.required_gpus]
            running_task = _spawn_ssim_task(
                task=task,
                repo_root=repo_root,
                assigned_gpu_ids=assigned_gpu_ids,
                log_dir=log_dir,
                task_index=task.task_id,
                pytest_extra_args=pytest_extra_args,
            )
            print(f"Started {task.test_name} on GPUs {','.join(assigned_gpu_ids)}")
            running_tasks.append(running_task)

        completed_tasks = []
        for running_task in running_tasks:
            returncode = running_task.process.poll()
            if returncode is not None:
                completed_tasks.append((running_task, returncode))

        for running_task, returncode in completed_tasks:
            _finalize_running_task(
                running_task=running_task,
                returncode=returncode,
                status="passed" if returncode == 0 else "failed",
                results=results,
                available_gpu_ids=available_gpu_ids,
                gpu_order=gpu_order,
            )
            running_tasks.remove(running_task)
            print(f"Finished {running_task.task.test_name} with exit code {returncode}")
            if returncode != 0 and fail_fast and not fail_fast_triggered:
                fail_fast_triggered = True

        if fail_fast_triggered and running_tasks:
            print("Fail-fast triggered: terminating active SSIM tasks.")
            _terminate_running_tasks(
                running_tasks=running_tasks,
                results=results,
                available_gpu_ids=available_gpu_ids,
                gpu_order=gpu_order,
            )

        if not completed_tasks and not fail_fast_triggered and running_tasks:
            time.sleep(1)

        if not running_tasks and fail_fast_triggered:
            break

    if fail_fast_triggered:
        for task in pending_tasks:
            results[task.task_id] = _TaskResult(
                task=task,
                status="skipped",
                returncode=-1,
                gpu_ids=[],
                log_path=None,
            )
    return results


def _collect_task_summaries(
    tasks: list[SSIMTask],
    results: dict[int, _TaskResult],
) -> list[_TaskSummary]:
    """Build serializable summaries, reading logs for failures."""
    summaries = []
    for task in tasks:
        result = results[task.task_id]
        log_content = None
        if result.status == "failed" and result.log_path and os.path.exists(result.log_path):
            with open(result.log_path, encoding="utf-8") as f:
                log_content = f.read()
        summaries.append(
            _TaskSummary(
                test_name=task.test_name,
                required_gpus=task.required_gpus,
                status=result.status,
                returncode=result.returncode,
                log_content=log_content,
            )
        )
    return summaries


def _print_combined_results(
    partition_results: list[_PartitionResult | None],
) -> int:
    """Print a unified report across all partitions."""
    passed = []
    failed = []
    terminated = []
    skipped = []
    first_failure = None

    for result in partition_results:
        if result is None:
            continue
        for s in result.task_summaries:
            label = f"[P{result.partition_index}] {s.test_name}"
            if s.status == "passed":
                passed.append(label)
            elif s.status == "failed":
                failed.append(label)
                if first_failure is None:
                    first_failure = (result.partition_index, s)
            elif s.status == "terminated":
                terminated.append(label)
            elif s.status == "skipped":
                skipped.append(label)

    if first_failure is not None:
        pi, s = first_failure
        print(f"\n{'=' * 60}")
        print(f"Partition: {pi}")
        print(f"Task: {s.test_name}")
        print(f"GPUs: {s.required_gpus}")
        print(f"Status: {s.status}")
        print(f"Exit code: {s.returncode}")
        print(f"{'=' * 60}")
        print(s.log_content or "No log output.")

    print("\nSSIM summary:")
    for i, result in enumerate(partition_results):
        if result is None:
            count = 0
            status = "cancelled"
        else:
            count = len(result.task_summaries)
            status = "passed" if result.exit_code == 0 else "failed"
        print(f"  Partition {i}: {status} ({count} tasks)")
    print(f"  passed: {len(passed)}")
    print(f"  failed: {len(failed)}")
    print(f"  terminated: {len(terminated)}")
    print(f"  skipped: {len(skipped)}")

    if failed:
        print(f"Failed: {', '.join(failed)}")
    if terminated:
        print(f"Terminated: {', '.join(terminated)}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")

    has_failures = bool(failed or terminated or skipped or any(r is None for r in partition_results))
    return 1 if has_failures else 0


@app.function(gpu=f"L40S:{SSIM_NUM_GPUS}", **SSIM_COMMON_KWARGS)
def run_ssim_partition(
    partition_index: int,
    num_partitions: int,
    git_repo: str,
    git_commit: str,
    pr_number: str = "false",
    hf_api_key: str = "",
    test_files_csv: str = "",
    model_ids_csv: str = "",
    ssim_full_quality: bool = False,
    ssim_reference_repo: str = "",
    skip_ssim_reference_download: bool = False,
    pytest_k: str = "",
    sync_generated_to_volume: bool = False,
    generated_volume_subdir: str = "",
    fail_fast: bool = True,
) -> _PartitionResult:
    selected_test_files = _split_csv_values(test_files_csv)
    selected_model_ids = _split_csv_values(model_ids_csv)
    repo_root, tasks = _prepare_ssim_workspace(
        git_repo=git_repo,
        git_commit=git_commit,
        pr_number=pr_number,
        hf_api_key=hf_api_key,
        selected_test_files=selected_test_files,
        selected_model_ids=selected_model_ids,
    )
    partition = _partition_tasks(tasks, partition_index, num_partitions)
    if not partition:
        print(f"Partition {partition_index}: no tasks assigned.")
        return _PartitionResult(
            partition_index=partition_index,
            task_summaries=[],
            exit_code=0,
        )
    print(f"Partition {partition_index}: running {len(partition)}/{len(tasks)} tasks")
    pytest_extra_args = _build_pytest_extra_args(
        ssim_full_quality=ssim_full_quality,
        ssim_reference_repo=ssim_reference_repo,
        skip_ssim_reference_download=skip_ssim_reference_download,
        pytest_k=pytest_k,
    )
    results = _schedule_ssim_tasks(
        repo_root,
        partition,
        pytest_extra_args=pytest_extra_args,
        fail_fast=fail_fast,
    )
    summaries = _collect_task_summaries(partition, results)
    has_failures = any(s.status != "passed" for s in summaries)
    if sync_generated_to_volume:
        quality_tier = _resolve_output_quality_tier(ssim_full_quality)
        resolved_subdir = _resolve_generated_volume_subdir(
            generated_volume_subdir,
            git_commit,
        )
        _sync_generated_videos_to_volume(
            repo_root,
            resolved_subdir,
            quality_tier,
        )
    return _PartitionResult(
        partition_index=partition_index,
        task_summaries=summaries,
        exit_code=1 if has_failures else 0,
    )


NUM_PARTITIONS = 2


@app.local_entrypoint()
def run_ssim_tests(
    git_repo: str = "",
    git_commit: str = "",
    pr_number: str = "",
    hf_api_key: str = "",
    test_files: str = "",
    model_ids: str = "",
    full_quality: bool = False,
    reference_repo: str = "",
    skip_reference_download: bool = False,
    pytest_k: str = "",
    sync_generated_to_volume: bool = False,
    generated_volume_subdir: str = "",
    no_fail_fast: bool = False,
):
    resolved_git_repo = _resolve_git_repo(git_repo)
    resolved_git_commit = _resolve_git_commit(git_commit)
    resolved_pr_number = _resolve_pull_request(pr_number)
    resolved_hf_api_key = _resolve_hf_api_key(hf_api_key)

    print(f"Running SSIM on repo: {resolved_git_repo}")
    print(f"Using commit: {resolved_git_commit}")
    if resolved_pr_number and resolved_pr_number != "false":
        print(f"Using PR ref: {resolved_pr_number}")
    if test_files.strip():
        print(f"Selected test files: {test_files}")
    if model_ids.strip():
        print(f"Selected model ids: {model_ids}")
    if pytest_k.strip():
        print(f"Using pytest -k filter: {pytest_k}")
    quality_tier = _resolve_output_quality_tier(full_quality)
    if sync_generated_to_volume:
        resolved_subdir = _resolve_generated_volume_subdir(
            generated_volume_subdir,
            resolved_git_commit,
        )
        print(
            "Raw generated videos will be saved to Modal volume path: "
            f"{_build_generated_volume_relative_path(generated_volume_subdir=resolved_subdir, quality_tier=quality_tier)}"
        )
    else:
        resolved_subdir = ""

    common_kwargs = dict(
        git_repo=resolved_git_repo,
        git_commit=resolved_git_commit,
        pr_number=resolved_pr_number,
        hf_api_key=resolved_hf_api_key,
        test_files_csv=test_files,
        model_ids_csv=model_ids,
        ssim_full_quality=full_quality,
        ssim_reference_repo=reference_repo,
        skip_ssim_reference_download=skip_reference_download,
        pytest_k=pytest_k,
        sync_generated_to_volume=sync_generated_to_volume,
        generated_volume_subdir=resolved_subdir,
        fail_fast=not no_fail_fast,
    )
    futures = [
        run_ssim_partition.spawn(
            partition_index=i,
            num_partitions=NUM_PARTITIONS,
            **common_kwargs,
        )
        for i in range(NUM_PARTITIONS)
    ]
    results: list[_PartitionResult | None] = [f.get() for f in futures]

    exit_code = _print_combined_results(results)
    if sync_generated_to_volume:
        download_src = _build_generated_volume_relative_path(
            generated_volume_subdir=resolved_subdir,
            quality_tier=quality_tier,
        )
        local_download_dir = _build_local_generated_download_dir(quality_tier)
        print(
            "To download raw generated videos locally, run:\n"
            f"  modal volume get hf-model-weights {download_src} "
            f"{local_download_dir}"
        )
        _print_local_reference_copy_command(quality_tier)
    if exit_code != 0:
        sys.exit(exit_code)
    print("All SSIM tasks passed.")
