# SPDX-License-Identifier: Apache-2.0
"""Run the SSIM suite across the GPUs assigned to one Slurm CI lane.

The Buildkite host grants this process an isolated four-GPU lease. This
scheduler discovers each test's ``REQUIRED_GPUS`` declaration and optional
``*_MODEL_TO_PARAMS`` dictionaries without importing the test modules, then
packs independent pytest processes onto the lease. The first failure stops
new work and terminates active siblings.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_GPUS = 4
MASTER_PORT_RANGE_SIZE = 100
TERMINATE_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class SSIMTask:
    task_id: int
    test_file: str
    required_gpus: int
    model_id: str | None = None

    @property
    def name(self) -> str:
        suffix = f"::{self.model_id}" if self.model_id else ""
        return f"{Path(self.test_file).name}{suffix}"

    @property
    def sort_key(self) -> tuple[str, str]:
        return (Path(self.test_file).name, self.model_id or "")


@dataclass
class RunningTask:
    task: SSIMTask
    process: subprocess.Popen[str]
    gpu_ids: list[str]
    log_path: Path
    log_handle: object


def extract_required_gpus(path: Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^REQUIRED_GPUS\s*=\s*(\d+)", line)
        if match:
            return int(match.group(1))
    return 1


def extract_model_ids(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    model_ids: list[str] = []
    for node in module.body:
        target_names: list[str] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            value = node.value
        if not any(name.endswith("MODEL_TO_PARAMS") for name in target_names) or not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                model_ids.append(key.value)
    return list(dict.fromkeys(model_ids))


def discover_tasks(ssim_dir: Path, test_files: Sequence[str] | None = None) -> list[SSIMTask]:
    paths = sorted(ssim_dir.glob("test_*.py"))
    if test_files:
        requested = set(test_files)
        invalid = sorted(name for name in requested if not re.fullmatch(r"test_[a-z0-9_]+\.py", name))
        if invalid:
            raise ValueError(f"Invalid SSIM test file selection: {invalid}")
        available = {path.name: path for path in paths}
        missing = sorted(requested - set(available))
        if missing:
            raise ValueError(f"Selected SSIM test files do not exist: {missing}")
        paths = [path for path in paths if path.name in requested]

    tasks: list[SSIMTask] = []
    for path in paths:
        required_gpus = extract_required_gpus(path)
        if not 1 <= required_gpus <= MAX_GPUS:
            raise ValueError(f"{path} requests {required_gpus} GPUs; supported range is 1-{MAX_GPUS}")
        model_ids = extract_model_ids(path)
        if not model_ids:
            model_ids = [None]
        for model_id in model_ids:
            tasks.append(
                SSIMTask(
                    task_id=len(tasks),
                    test_file=f"./fastvideo/tests/ssim/{path.name}",
                    required_gpus=required_gpus,
                    model_id=model_id,
                ))
    return sorted(tasks, key=lambda task: task.sort_key)


def visible_gpu_ids() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw:
        return [str(index) for index in range(MAX_GPUS)]
    gpu_ids = list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not gpu_ids:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not contain any GPU ids")
    return gpu_ids


def task_master_port(task_id: int, base_port: str | None = None) -> str:
    """Return a task-specific port inside the runner-assigned lane range."""
    if not 0 <= task_id < MASTER_PORT_RANGE_SIZE:
        raise ValueError(f"SSIM task id is outside the runner-assigned 100-port range: {task_id}")
    try:
        port = int(base_port or "29500") + task_id
    except ValueError as error:
        raise ValueError(f"Invalid MASTER_PORT: {base_port!r}") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"SSIM task master port is outside 1-65535: {port}")
    return str(port)


def spawn_task(task: SSIMTask, gpu_ids: list[str], log_dir: Path, extra_args: list[str]) -> RunningTask:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.name)
    log_path = log_dir / f"{task.task_id:03d}_{safe_name}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = task_master_port(task.task_id, env.get("MASTER_PORT"))
    if task.model_id:
        env["FASTVIDEO_SSIM_MODEL_ID"] = task.model_id
    else:
        env.pop("FASTVIDEO_SSIM_MODEL_ID", None)
    process = subprocess.Popen(
        ["pytest", task.test_file, "-vs", *extra_args],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    return RunningTask(task, process, gpu_ids, log_path, log_handle)


def terminate(tasks: list[RunningTask]) -> None:
    for running in tasks:
        if running.process.poll() is None:
            try:
                os.killpg(running.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + TERMINATE_TIMEOUT_SECONDS
    while time.monotonic() < deadline and any(task.process.poll() is None for task in tasks):
        time.sleep(1)
    for running in tasks:
        if running.process.poll() is None:
            try:
                os.killpg(running.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run(tasks: list[SSIMTask], extra_args: list[str]) -> int:
    available = visible_gpu_ids()
    if max((task.required_gpus for task in tasks), default=0) > len(available):
        raise RuntimeError(f"SSIM tasks require more GPUs than the {len(available)} visible devices")
    order = {gpu_id: index for index, gpu_id in enumerate(available)}
    pending = list(tasks)
    running: list[RunningTask] = []
    results: dict[int, tuple[str, int, Path | None]] = {}
    log_dir = Path(tempfile.mkdtemp(prefix="fastvideo-ssim-logs-"))
    failed = False

    try:
        while pending or running:
            while not failed:
                next_index = next(
                    (index for index, task in enumerate(pending) if task.required_gpus <= len(available)),
                    None,
                )
                if next_index is None:
                    break
                task = pending.pop(next_index)
                assigned = available[:task.required_gpus]
                del available[:task.required_gpus]
                running_task = spawn_task(task, assigned, log_dir, extra_args)
                running.append(running_task)
                print(f"Started {task.name} on GPUs {','.join(assigned)}", flush=True)

            completed = [task for task in running if task.process.poll() is not None]
            for running_task in completed:
                returncode = running_task.process.wait()
                running_task.log_handle.close()
                available.extend(running_task.gpu_ids)
                available.sort(key=order.__getitem__)
                status = "passed" if returncode == 0 else "failed"
                results[running_task.task.task_id] = (status, returncode, running_task.log_path)
                running.remove(running_task)
                print(f"Finished {running_task.task.name} with exit code {returncode}", flush=True)
                failed = failed or returncode != 0

            if failed and running:
                print("Fail-fast triggered: terminating active SSIM tasks.", flush=True)
                terminate(running)
                for running_task in running:
                    returncode = running_task.process.wait()
                    running_task.log_handle.close()
                    results[running_task.task.task_id] = ("terminated", returncode, running_task.log_path)
                running.clear()
            elif running and not completed:
                time.sleep(1)
            elif failed:
                break
    except BaseException:
        terminate(running)
        raise

    for task in pending:
        results[task.task_id] = ("skipped", -1, None)

    print("\nSSIM summary:")
    counts = {"passed": 0, "failed": 0, "terminated": 0, "skipped": 0}
    for task in tasks:
        status, returncode, log_path = results[task.task_id]
        counts[status] += 1
        print(f"  {status:10} {task.name} (gpus={task.required_gpus}, rc={returncode})")
        if status == "failed" and log_path:
            print(f"\n--- {task.name} failure log ---")
            print(log_path.read_text(encoding="utf-8", errors="replace"))
    print("  " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    return 1 if any(counts[name] for name in ("failed", "terminated", "skipped")) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-quality", action="store_true")
    parser.add_argument("--reference-repo", default="")
    parser.add_argument("--skip-reference-download", action="store_true")
    parser.add_argument("--bootstrap-mode", action="store_true")
    parser.add_argument(
        "--test-file",
        action="append",
        default=[],
        help="Run one SSIM test basename; repeat for a focused merge gate.",
    )
    parser.add_argument("-k", dest="pytest_k", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pytest_args: list[str] = []
    if args.full_quality:
        pytest_args.append("--ssim-full-quality")
    if args.reference_repo:
        pytest_args.extend(["--ssim-reference-repo", args.reference_repo])
    if args.skip_reference_download:
        pytest_args.append("--skip-ssim-reference-download")
    if args.bootstrap_mode:
        pytest_args.append("--ssim-bootstrap-mode")
    if args.pytest_k:
        pytest_args.extend(["-k", args.pytest_k])
    tasks = discover_tasks(Path("fastvideo/tests/ssim"), args.test_file)
    if not tasks:
        raise RuntimeError("No SSIM tests discovered")
    print(f"Discovered {len(tasks)} SSIM tasks across {len(visible_gpu_ids())} GPUs")
    return run(tasks, pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
