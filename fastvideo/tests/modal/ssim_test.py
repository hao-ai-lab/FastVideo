import ast
import glob
import os
import re
from dataclasses import dataclass
from typing import Any

import modal

app = modal.App()

model_vol = modal.Volume.from_name("hf-model-weights")
image_version = os.getenv("IMAGE_VERSION")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
print(f"Using image: {image_tag}")

image = (
    modal.Image.from_registry(image_tag, add_python="3.12")
    .run_commands("rm -rf /FastVideo")
    .apt_install(
        "cmake",
        "pkg-config",
        "build-essential",
        "curl",
        "libssl-dev",
        "ffmpeg",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf "
        "https://sh.rustup.rs | sh -s -- -y --default-toolchain stable"
    )
    .run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc")
    .env({
        "PATH": "/root/.cargo/bin:$PATH",
        "BUILDKITE_REPO": os.environ.get("BUILDKITE_REPO", ""),
        "BUILDKITE_COMMIT": os.environ.get("BUILDKITE_COMMIT", ""),
        "BUILDKITE_PULL_REQUEST": os.environ.get(
            "BUILDKITE_PULL_REQUEST", ""
        ),
        "IMAGE_VERSION": os.environ.get("IMAGE_VERSION", ""),
    })
)

SSIM_NUM_GPUS = 8
SSIM_TERMINATE_TIMEOUT_S = 30
SSIM_COMMON_KWARGS = dict(
    image=image,
    timeout=5400,
    secrets=[modal.Secret.from_dict(
        {"HF_API_KEY": os.environ.get("HF_API_KEY", "")}
    )],
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
    gpu_ids: list[int]
    log_path: str
    log_handle: Any


@dataclass
class _TaskResult:
    task: SSIMTask
    status: str
    returncode: int
    gpu_ids: list[int]
    log_path: str | None = None


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
            target_names = [
                target.id for target in node.targets
                if isinstance(target, ast.Name)
            ]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target_names = [node.target.id]
            value_node = node.value

        if not target_names or value_node is None:
            continue
        if not any(name.endswith("MODEL_TO_PARAMS")
                   for name in target_names):
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


def _discover_ssim_tasks(ssim_dir: str) -> list[SSIMTask]:
    tasks = []
    task_id = 0
    test_files = sorted(glob.glob(os.path.join(ssim_dir, "test_*.py")))
    for filepath in test_files:
        required_gpus = _extract_required_gpus(filepath)
        if required_gpus < 1 or required_gpus > SSIM_NUM_GPUS:
            raise ValueError(
                f"{filepath} requires {required_gpus} GPUs, "
                f"but scheduler supports up to {SSIM_NUM_GPUS}."
            )
        rel_path = f"./fastvideo/tests/ssim/{os.path.basename(filepath)}"
        model_ids = _extract_model_ids(filepath)
        if model_ids:
            for model_id in model_ids:
                tasks.append(SSIMTask(
                    task_id=task_id,
                    test_file=rel_path,
                    required_gpus=required_gpus,
                    model_id=model_id,
                ))
                task_id += 1
        else:
            tasks.append(SSIMTask(
                task_id=task_id,
                test_file=rel_path,
                required_gpus=required_gpus,
            ))
            task_id += 1
    return sorted(tasks, key=lambda task: task.sort_key)


def _build_checkout_command(git_commit: str, pr_number: str | None) -> str:
    import shlex

    if pr_number and pr_number != "false":
        try:
            pr_id = int(pr_number)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid BUILDKITE_PULL_REQUEST value: {pr_number}"
            ) from error
        return (
            "git fetch --prune origin "
            f"refs/pull/{pr_id}/head && "
            "git checkout FETCH_HEAD"
        )
    return f"git checkout {shlex.quote(git_commit)}"


def _prepare_ssim_workspace() -> tuple[str, list[SSIMTask]]:
    import shlex
    import subprocess

    git_repo = os.environ.get("BUILDKITE_REPO")
    git_commit = os.environ.get("BUILDKITE_COMMIT")
    pr_number = os.environ.get("BUILDKITE_PULL_REQUEST")
    if not git_repo or not git_commit:
        raise RuntimeError("BUILDKITE_REPO and BUILDKITE_COMMIT must be set.")

    checkout_command = _build_checkout_command(git_commit, pr_number)
    repo_root = "/FastVideo"

    command = f"""
    set -euo pipefail
    source $HOME/.local/bin/env
    source /opt/venv/bin/activate
    rm -rf {shlex.quote(repo_root)}
    git clone {shlex.quote(git_repo)} {shlex.quote(repo_root)}
    cd {shlex.quote(repo_root)}
    {checkout_command}
    git submodule update --init --recursive
    cd fastvideo-kernel
    ./build.sh
    cd ..
    uv pip install -e .[test]
    export HF_HOME='/root/data/.cache'
    hf auth login --token "$HF_API_KEY"
    """
    subprocess.run(
        ["/bin/bash", "-lc", command],
        check=True,
    )

    ssim_dir = os.path.join(repo_root, "fastvideo", "tests", "ssim")
    tasks = _discover_ssim_tasks(ssim_dir)
    if not tasks:
        raise RuntimeError("No SSIM test files found.")
    return repo_root, tasks


def _spawn_ssim_task(
    task: SSIMTask,
    repo_root: str,
    assigned_gpu_ids: list[int],
    log_dir: str,
    task_index: int,
) -> _RunningTask:
    import shlex
    import subprocess

    safe_test_name = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", task.test_name
    )
    log_path = os.path.join(
        log_dir, f"{task_index:03d}_{safe_test_name}.log"
    )
    command = (
        "set -euo pipefail && "
        "source $HOME/.local/bin/env && "
        "source /opt/venv/bin/activate && "
        f"pytest {shlex.quote(task.test_file)} -vs"
    )
    env = os.environ.copy()
    env["HF_HOME"] = "/root/data/.cache"
    # MultiprocExecutor returns CUDA tensors through mp pipes (CUDA IPC).
    # On kernels without pidfd_open support, PyTorch fails when
    # expandable_segments=True. Force False for CI compatibility.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu_id) for gpu_id in assigned_gpu_ids
    )
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
    available_gpu_ids: list[int],
) -> None:
    running_task.log_handle.close()
    available_gpu_ids.extend(running_task.gpu_ids)
    available_gpu_ids.sort()
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
    available_gpu_ids: list[int],
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
        )
        running_tasks.remove(running_task)


def _schedule_ssim_tasks(
    repo_root: str,
    tasks: list[SSIMTask],
) -> dict[int, _TaskResult]:
    import tempfile
    import time

    pending_tasks = list(tasks)
    running_tasks = []
    available_gpu_ids = list(range(SSIM_NUM_GPUS))
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
            assigned_gpu_ids = available_gpu_ids[:task.required_gpus]
            del available_gpu_ids[:task.required_gpus]
            running_task = _spawn_ssim_task(
                task=task,
                repo_root=repo_root,
                assigned_gpu_ids=assigned_gpu_ids,
                log_dir=log_dir,
                task_index=task.task_id,
            )
            print(
                f"Started {task.test_name} on GPUs "
                f"{','.join(str(idx) for idx in assigned_gpu_ids)}"
            )
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
            )
            running_tasks.remove(running_task)
            print(
                f"Finished {running_task.task.test_name} with "
                f"exit code {returncode}"
            )
            if returncode != 0 and not fail_fast_triggered:
                fail_fast_triggered = True

        if fail_fast_triggered and running_tasks:
            print("Fail-fast triggered: terminating active SSIM tasks.")
            _terminate_running_tasks(
                running_tasks=running_tasks,
                results=results,
                available_gpu_ids=available_gpu_ids,
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


def _print_ssim_task_results(
    tasks: list[SSIMTask],
    results: dict[int, _TaskResult],
) -> int:
    failed = []
    terminated = []
    skipped = []
    passed = []
    failed_task_results = []

    for task in tasks:
        result = results[task.task_id]
        if result.status == "passed":
            passed.append(task.test_name)
        elif result.status == "failed":
            failed.append(task.test_name)
            failed_task_results.append((task, result))
        elif result.status == "terminated":
            terminated.append(task.test_name)
            failed_task_results.append((task, result))
        elif result.status == "skipped":
            skipped.append(task.test_name)

    for task, result in failed_task_results:
        print(f"\n{'=' * 60}")
        print(f"Task: {task.test_name}")
        print(f"GPUs: {task.required_gpus}")
        print(f"Status: {result.status}")
        print(f"Exit code: {result.returncode}")
        print(f"{'=' * 60}")
        if result.log_path and os.path.exists(result.log_path):
            with open(result.log_path, encoding="utf-8") as log_file:
                print(log_file.read())
        else:
            print("No log output.")

    print("\nSSIM summary:")
    print(f"  passed: {len(passed)}")
    print(f"  failed: {len(failed)}")
    print(f"  terminated: {len(terminated)}")
    print(f"  skipped: {len(skipped)}")

    if failed:
        print(f"Failed tasks: {', '.join(failed)}")
    if terminated:
        print(f"Terminated tasks: {', '.join(terminated)}")
    if skipped:
        print(f"Skipped tasks: {', '.join(skipped)}")

    return 1 if failed or terminated or skipped else 0


@app.function(gpu="L40S:8", **SSIM_COMMON_KWARGS)
def run_ssim_tests_single_instance() -> int:
    repo_root, tasks = _prepare_ssim_workspace()
    results = _schedule_ssim_tasks(repo_root, tasks)
    return _print_ssim_task_results(tasks, results)


@app.local_entrypoint()
def run_ssim_tests():
    import sys

    exit_code = run_ssim_tests_single_instance.remote()
    if exit_code != 0:
        sys.exit(exit_code)
    print("All SSIM tasks passed.")
