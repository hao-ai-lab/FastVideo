# SPDX-License-Identifier: Apache-2.0
"""Run SSIM tests locally with exclusive, mixed-size GPU allocations.

The SSIM suite contains one-, two-, and four-GPU tests. Plain pytest runs
them serially, while pytest-xdist cannot safely infer how many GPUs each test
needs. This runner mirrors the Modal scheduler's static discovery and assigns
disjoint ``CUDA_VISIBLE_DEVICES`` sets to concurrent pytest subprocesses.

Example:

.. code-block:: bash

    .venv/bin/python fastvideo/tests/ssim/run_parallel.py \
        --ssim-full-quality \
        --hf-home /mnt/.cache \
        --gpu-ids 0,1,2,3

Non-zero pytest exits do not stop other tasks by default. This is important
for visual-generation audits where a missing device-specific reference may
fail after the generated artifact has already been saved.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import resource
import signal
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SSIM_DIR = Path(__file__).resolve().parent
DEFAULT_HF_HOME = Path("/mnt/.cache")
DEFAULT_ALLOC_CONF = "expandable_segments:False"
HF_TOKEN_ENV_KEYS = ("HF_API_KEY", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")
HANDLED_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


def _disable_core_dumps() -> None:
    """Prevent routine worker failures from writing multi-gigabyte core files."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    """Route termination signals through the normal worker cleanup path."""
    raise KeyboardInterrupt


def _unblock_task_signals() -> None:
    """Reset the launcher's short-lived blocked mask before child exec."""
    signal.pthread_sigmask(signal.SIG_UNBLOCK, HANDLED_SIGNALS)


@dataclass(frozen=True)
class SSIMTask:
    task_id: int
    test_file: Path
    required_gpus: int
    model_id: str | None = None
    test_function: str | None = None

    @property
    def node_id(self) -> str:
        relative_file = self.test_file.relative_to(REPO_ROOT).as_posix()
        if self.test_function is None:
            return relative_file
        return f"{relative_file}::{self.test_function}"

    @property
    def display_name(self) -> str:
        parts = [self.test_file.name]
        if self.test_function is not None:
            parts.append(self.test_function)
        if self.model_id is not None:
            parts.append(self.model_id)
        return "::".join(parts)

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return (
            self.test_file.name,
            self.test_function or "",
            self.model_id or "",
        )


@dataclass
class RunningTask:
    task: SSIMTask
    process: subprocess.Popen[str]
    gpu_ids: list[str]
    log_path: Path
    junit_path: Path
    log_handle: Any
    started_at: float
    command: list[str]


@dataclass
class TaskResult:
    task_id: int
    test_name: str
    node_id: str
    model_id: str | None
    required_gpus: int
    gpu_ids: list[str]
    status: str
    returncode: int | None
    duration_seconds: float
    tests: int
    passed: int
    failures: int
    errors: int
    skipped: int
    log_path: str
    junit_path: str
    command: list[str]


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            return [node.target.id]
        return []
    return [target.id for target in node.targets if isinstance(target, ast.Name)]


def _extract_required_gpus(module_ast: ast.Module, filepath: Path) -> int:
    for node in module_ast.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if "REQUIRED_GPUS" not in _assignment_names(node):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            required_gpus = node.value.value
            if required_gpus < 1:
                raise ValueError(f"{filepath}: REQUIRED_GPUS must be positive")
            return required_gpus
        raise ValueError(f"{filepath}: REQUIRED_GPUS must be an integer literal")
    return 1


def _extract_string_constants(module_ast: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in module_ast.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for name in _assignment_names(node):
            constants[name] = node.value.value
    return constants


def _extract_model_ids(module_ast: ast.Module) -> list[str]:
    """Extract literal model IDs from all ``*_MODEL_TO_PARAMS`` maps.

    Both default and full-quality maps are inspected and de-duplicated. Named
    string constants are supported as dictionary keys (for example SD3.5's
    ``MODEL_ID``), fixing a limitation in the current Modal discovery helper.
    """
    string_constants = _extract_string_constants(module_ast)
    model_ids: list[str] = []
    seen: set[str] = set()
    for node in module_ast.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not any(name.endswith("MODEL_TO_PARAMS") for name in _assignment_names(node)):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            model_id: str | None = None
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                model_id = key.value
            elif isinstance(key, ast.Name):
                model_id = string_constants.get(key.id)
            if model_id is not None and model_id not in seen:
                seen.add(model_id)
                model_ids.append(model_id)
    return model_ids


def _extract_test_functions(module_ast: ast.Module) -> list[str]:
    return [
        node.name for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]


def _normalize_selected_files(values: list[str]) -> set[str]:
    selected: set[str] = set()
    for value in values:
        for part in value.split(","):
            cleaned = part.strip()
            if cleaned:
                selected.add(Path(cleaned).name)
    return selected


def _normalize_selected_models(values: list[str]) -> set[str]:
    selected: set[str] = set()
    for value in values:
        selected.update(part.strip() for part in value.split(",") if part.strip())
    return selected


def discover_tasks(
    *,
    selected_test_files: set[str] | None = None,
    selected_model_ids: set[str] | None = None,
) -> list[SSIMTask]:
    selected_test_files = selected_test_files or set()
    selected_model_ids = selected_model_ids or set()
    matched_files: set[str] = set()
    matched_models: set[str] = set()
    task_specs: list[tuple[Path, int, str | None, str | None]] = []

    for filepath in sorted(SSIM_DIR.glob("test_*_similarity.py")):
        if selected_test_files and filepath.name not in selected_test_files:
            continue
        matched_files.add(filepath.name)
        module_ast = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
        required_gpus = _extract_required_gpus(module_ast, filepath)
        model_ids = _extract_model_ids(module_ast)

        if model_ids:
            for model_id in model_ids:
                if selected_model_ids and model_id not in selected_model_ids:
                    continue
                matched_models.add(model_id)
                task_specs.append((filepath, required_gpus, model_id, None))
            continue

        if selected_model_ids:
            continue
        test_functions = _extract_test_functions(module_ast)
        if not test_functions:
            task_specs.append((filepath, required_gpus, None, None))
        else:
            # Function-level splitting gives independent LongCat T2V/I2V/VC
            # tasks, keeping three one-GPU workers busy instead of serializing
            # the whole file on one GPU.
            for test_function in test_functions:
                task_specs.append((filepath, required_gpus, None, test_function))

    missing_files = sorted(selected_test_files - matched_files)
    if missing_files:
        raise ValueError(f"Requested SSIM test file(s) not found: {', '.join(missing_files)}")
    missing_models = sorted(selected_model_ids - matched_models)
    if missing_models:
        raise ValueError(f"Requested SSIM model ID(s) not found: {', '.join(missing_models)}")

    task_specs.sort(key=lambda spec: (spec[0].name, spec[3] or "", spec[2] or ""))
    return [
        SSIMTask(
            task_id=index,
            test_file=filepath,
            required_gpus=required_gpus,
            model_id=model_id,
            test_function=test_function,
        )
        for index, (filepath, required_gpus, model_id, test_function) in enumerate(task_specs)
    ]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _parse_gpu_ids(value: str) -> list[str]:
    gpu_ids: list[str] = []
    seen: set[str] = set()
    for item in value.split(","):
        gpu_id = item.strip()
        if not gpu_id or gpu_id in seen:
            continue
        seen.add(gpu_id)
        gpu_ids.append(gpu_id)
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU ID")
    return gpu_ids


def _preflight_gpus(gpu_ids: list[str]) -> list[str]:
    probe = (
        "import json, torch\n"
        "names=[]\n"
        "for i in range(torch.cuda.device_count()):\n"
        "    names.append(torch.cuda.get_device_name(i))\n"
        "    torch.empty(1, device=f'cuda:{i}')\n"
        "torch.cuda.synchronize() if names else None\n"
        "print(json.dumps({'count': torch.cuda.device_count(), 'names': names}))\n"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"CUDA preflight failed: {detail or f'exit code {result.returncode}'}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"CUDA preflight returned invalid output: {result.stdout!r}") from error
    if payload.get("count") != len(gpu_ids):
        recovery_hint = ""
        if payload.get("count", 0) == 0:
            recovery_hint = (
                " No pytest task was launched. Verify `nvidia-smi` and recreate "
                "the container/workspace with NVIDIA GPU device grants if it "
                "reports `Failed to initialize NVML`."
            )
        raise RuntimeError(
            f"Requested GPU IDs {gpu_ids}, but PyTorch can initialize only "
            f"{payload.get('count', 0)} device(s). stderr: {result.stderr.strip()}"
            f"{recovery_hint}"
        )
    return [str(name) for name in payload.get("names", [])]


def _read_junit_counts(junit_path: Path) -> dict[str, int]:
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "passed": 0}
    if not junit_path.exists():
        return counts
    try:
        root = ET.parse(junit_path).getroot()
    except (ET.ParseError, OSError):
        return counts

    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    for suite in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            counts[key] += int(suite.attrib.get(key, 0))
    counts["passed"] = max(0, counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"])
    return counts


def _classify_status(returncode: int, counts: dict[str, int]) -> str:
    if returncode != 0:
        return "failed"
    if counts["tests"] > 0 and counts["skipped"] == counts["tests"]:
        return "skipped"
    return "passed"


def _build_command(
    task: SSIMTask,
    *,
    junit_path: Path,
    full_quality: bool,
    skip_reference_download: bool,
    pytest_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        task.node_id,
        "-vs",
        f"--junitxml={junit_path}",
        "-o",
        "junit_family=xunit2",
    ]
    if full_quality:
        command.append("--ssim-full-quality")
    if skip_reference_download:
        command.append("--skip-ssim-reference-download")
    command.extend(pytest_args)
    return command


def _launch_task(
    task: SSIMTask,
    *,
    gpu_ids: list[str],
    run_dir: Path,
    hf_home: Path,
    full_quality: bool,
    skip_reference_download: bool,
    pytest_args: list[str],
) -> RunningTask:
    safe_name = _safe_name(f"{task.task_id:03d}_{task.display_name}")
    log_path = run_dir / "logs" / f"{safe_name}.log"
    junit_path = run_dir / "junit" / f"{safe_name}.xml"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(
        task,
        junit_path=junit_path,
        full_quality=full_quality,
        skip_reference_download=skip_reference_download,
        pytest_args=pytest_args,
    )
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
        "HF_HOME": str(hf_home),
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_ALLOC_CONF),
        "TOKENIZERS_PARALLELISM": "false",
    })
    if full_quality:
        env["FASTVIDEO_SSIM_FULL_QUALITY"] = "1"
    else:
        env.pop("FASTVIDEO_SSIM_FULL_QUALITY", None)
    if task.model_id is None:
        env.pop("FASTVIDEO_SSIM_MODEL_ID", None)
    else:
        env["FASTVIDEO_SSIM_MODEL_ID"] = task.model_id

    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=_unblock_task_signals,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    return RunningTask(
        task=task,
        process=process,
        gpu_ids=gpu_ids,
        log_path=log_path,
        junit_path=junit_path,
        log_handle=log_handle,
        started_at=time.monotonic(),
        command=command,
    )


def _terminate_running(running: list[RunningTask], timeout_seconds: float = 30.0) -> None:
    for item in running:
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and any(item.process.poll() is None for item in running):
        time.sleep(0.25)
    for item in running:
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _finalize_task(item: RunningTask, returncode: int) -> TaskResult:
    item.log_handle.close()
    counts = _read_junit_counts(item.junit_path)
    return TaskResult(
        task_id=item.task.task_id,
        test_name=item.task.display_name,
        node_id=item.task.node_id,
        model_id=item.task.model_id,
        required_gpus=item.task.required_gpus,
        gpu_ids=item.gpu_ids,
        status=_classify_status(returncode, counts),
        returncode=returncode,
        duration_seconds=round(time.monotonic() - item.started_at, 3),
        tests=counts["tests"],
        passed=counts["passed"],
        failures=counts["failures"],
        errors=counts["errors"],
        skipped=counts["skipped"],
        log_path=str(item.log_path),
        junit_path=str(item.junit_path),
        command=item.command,
    )


def run_tasks(
    tasks: list[SSIMTask],
    *,
    gpu_ids: list[str],
    run_dir: Path,
    hf_home: Path,
    full_quality: bool,
    skip_reference_download: bool,
    pytest_args: list[str],
    fail_fast: bool,
) -> tuple[list[TaskResult], bool]:
    pending = list(tasks)
    running: list[RunningTask] = []
    results: list[TaskResult] = []
    free_gpu_ids = list(gpu_ids)
    gpu_order = {gpu_id: index for index, gpu_id in enumerate(gpu_ids)}
    interrupted = False
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in HANDLED_SIGNALS
    }
    for signum in HANDLED_SIGNALS:
        signal.signal(signum, _raise_keyboard_interrupt)

    try:
        while pending or running:
            while pending:
                next_index = next(
                    (index for index, task in enumerate(pending) if task.required_gpus <= len(free_gpu_ids)),
                    None,
                )
                if next_index is None:
                    break
                task = pending.pop(next_index)
                assigned = free_gpu_ids[:task.required_gpus]
                del free_gpu_ids[:task.required_gpus]
                previous_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK,
                    HANDLED_SIGNALS,
                )
                try:
                    item = _launch_task(
                        task,
                        gpu_ids=assigned,
                        run_dir=run_dir,
                        hf_home=hf_home,
                        full_quality=full_quality,
                        skip_reference_download=skip_reference_download,
                        pytest_args=pytest_args,
                    )
                    running.append(item)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                print(
                    f"START [{task.task_id:03d}] {task.display_name} "
                    f"on GPU(s) {','.join(assigned)} -> {item.log_path}",
                    flush=True,
                )

            completed = [item for item in running if item.process.poll() is not None]
            if not completed:
                time.sleep(0.5)
                continue

            stop_after_failure = False
            for item in completed:
                returncode = item.process.wait()
                result = _finalize_task(item, returncode)
                results.append(result)
                running.remove(item)
                free_gpu_ids.extend(item.gpu_ids)
                free_gpu_ids.sort(key=lambda gpu_id: gpu_order[gpu_id])
                print(
                    f"DONE  [{item.task.task_id:03d}] {item.task.display_name}: "
                    f"{result.status} rc={returncode} in {result.duration_seconds:.1f}s",
                    flush=True,
                )
                stop_after_failure = stop_after_failure or result.status == "failed"

            if stop_after_failure and fail_fast:
                print("Fail-fast enabled; terminating active tasks.", flush=True)
                _terminate_running(running)
                break
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; terminating active SSIM tasks.", flush=True)
        for signum in HANDLED_SIGNALS:
            signal.signal(signum, signal.SIG_IGN)
        _terminate_running(running)
    except BaseException:
        print("Runner error; terminating active SSIM tasks.", flush=True)
        for signum in HANDLED_SIGNALS:
            signal.signal(signum, signal.SIG_IGN)
        _terminate_running(running)
        raise
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
        for item in list(running):
            returncode = item.process.wait()
            results.append(_finalize_task(item, returncode))
        for task in pending:
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    test_name=task.display_name,
                    node_id=task.node_id,
                    model_id=task.model_id,
                    required_gpus=task.required_gpus,
                    gpu_ids=[],
                    status="not_run",
                    returncode=None,
                    duration_seconds=0.0,
                    tests=0,
                    passed=0,
                    failures=0,
                    errors=0,
                    skipped=0,
                    log_path="",
                    junit_path="",
                    command=[],
                )
            )
    results.sort(key=lambda result: result.task_id)
    return results, interrupted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-ids",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
        help="Comma-separated physical GPU IDs or UUIDs (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help=f"Shared Hugging Face cache root (default: {DEFAULT_HF_HOME}).",
    )
    parser.add_argument("--ssim-full-quality", action="store_true", help="Select full-quality SSIM parameters.")
    parser.add_argument(
        "--skip-reference-download",
        action="store_true",
        help="Pass --skip-ssim-reference-download to every pytest task.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed task.")
    parser.add_argument(
        "--allow-core-dumps",
        action="store_true",
        help="Keep the shell's core-dump limit instead of disabling worker core dumps.",
    )
    parser.add_argument("--test-file", action="append", default=[], help="Run only this test file (repeatable/CSV).")
    parser.add_argument("--model-id", action="append", default=[], help="Run only this model ID (repeatable/CSV).")
    parser.add_argument("--pytest-arg", action="append", default=[], help="Additional argument passed to pytest.")
    parser.add_argument("--run-dir", type=Path, help="Directory for task logs, JUnit XML, and summary.json.")
    parser.add_argument(
        "--keep-existing-generated",
        action="store_true",
        help="Do not archive existing generated artifacts before the run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovery/allocation requirements without probing GPUs or launching pytest.",
    )
    return parser


def _default_run_dir(full_quality: bool) -> Path:
    tier = "full_quality" if full_quality else "default"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return SSIM_DIR / "generated_videos" / tier / "_runs" / timestamp


def _archive_existing_generated(full_quality: bool, run_dir: Path) -> list[Path]:
    """Move stale generated artifacts aside before starting a new run.

    Video output helpers add numeric suffixes when a target already exists,
    while several tests later inspect the unsuffixed path. Archiving avoids a
    stale artifact being mistaken for the result of the current run.
    """
    tier = "full_quality" if full_quality else "default"
    tier_root = SSIM_DIR / "generated_videos" / tier
    if not tier_root.exists():
        return []

    archive_root = run_dir / "previous_generated"
    archived: list[Path] = []
    for path in sorted(tier_root.iterdir()):
        if path.name == "_runs":
            continue
        destination = archive_root / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        archived.append(destination)
    return archived


def _write_summary(
    path: Path,
    *,
    started_at: str,
    finished_at: str,
    hf_home: Path,
    gpu_ids: list[str],
    gpu_names: list[str],
    full_quality: bool,
    interrupted: bool,
    results: list[TaskResult],
) -> None:
    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "hf_home": str(hf_home),
        "gpu_ids": gpu_ids,
        "gpu_names": gpu_names,
        "full_quality": full_quality,
        "interrupted": interrupted,
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.allow_core_dumps:
        _disable_core_dumps()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    selected_files = _normalize_selected_files(args.test_file)
    selected_models = _normalize_selected_models(args.model_id)
    tasks = discover_tasks(
        selected_test_files=selected_files,
        selected_model_ids=selected_models,
    )
    if not tasks:
        raise RuntimeError("No SSIM tasks matched the requested filters")
    max_required = max(task.required_gpus for task in tasks)
    if max_required > len(gpu_ids):
        raise RuntimeError(
            f"A selected task requires {max_required} GPUs, but --gpu-ids provides only {len(gpu_ids)}"
        )

    print(f"Discovered {len(tasks)} SSIM task(s) across {len(gpu_ids)} GPU slot(s):", flush=True)
    for task in tasks:
        print(f"  [{task.task_id:03d}] {task.required_gpus} GPU(s)  {task.display_name}", flush=True)
    if args.dry_run:
        return 0

    hf_home = args.hf_home.expanduser().resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    if not any(os.environ.get(key, "").strip() for key in HF_TOKEN_ENV_KEYS):
        print(
            "WARNING: no Hugging Face token is set; gated model tasks may fail.",
            flush=True,
        )
    run_dir = (args.run_dir or _default_run_dir(args.ssim_full_quality)).expanduser().resolve()
    gpu_names = _preflight_gpus(gpu_ids)
    if not args.keep_existing_generated:
        archived = _archive_existing_generated(args.ssim_full_quality, run_dir)
        if archived:
            print(f"Archived {len(archived)} existing generated path(s) under {run_dir}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"CUDA preflight passed: {dict(zip(gpu_ids, gpu_names, strict=True))}", flush=True)
    print(f"HF_HOME={hf_home}", flush=True)
    print(f"Run artifacts: {run_dir}", flush=True)

    started_at = datetime.now(timezone.utc).isoformat()
    results, interrupted = run_tasks(
        tasks,
        gpu_ids=gpu_ids,
        run_dir=run_dir,
        hf_home=hf_home,
        full_quality=args.ssim_full_quality,
        skip_reference_download=args.skip_reference_download,
        pytest_args=args.pytest_arg,
        fail_fast=args.fail_fast,
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    summary_path = run_dir / "summary.json"
    _write_summary(
        summary_path,
        started_at=started_at,
        finished_at=finished_at,
        hf_home=hf_home,
        gpu_ids=gpu_ids,
        gpu_names=gpu_names,
        full_quality=args.ssim_full_quality,
        interrupted=interrupted,
        results=results,
    )

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    print(f"Summary: {status_counts}", flush=True)
    print(f"Detailed results: {summary_path}", flush=True)
    if interrupted:
        return 130
    return 1 if any(result.status in {"failed", "not_run"} for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
