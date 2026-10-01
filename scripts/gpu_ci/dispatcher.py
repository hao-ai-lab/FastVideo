"""Suite lifecycle and local-host recovery for opt-in Buildkite GPU backends."""

from __future__ import annotations

import fcntl
import hashlib
import json
import signal
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .admission import AdmissionStore

INFRASTRUCTURE_FAILURE = 97


def run_directory(config: dict[str, Any], build_id: str) -> Path:
    digest = hashlib.sha256(build_id.encode()).hexdigest()[:32]
    return Path(config["artifacts_dir"]) / digest


@contextmanager
def build_lock(directory: Path) -> Iterator[None]:
    """Serialize launch and recovery; never stop a resource during its creation."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "dispatcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A dispatcher is still running; cancel its Buildkite job first") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def make_store(config: dict[str, Any]) -> AdmissionStore:
    return AdmissionStore(config["state_path"], config.get("max_active_prs", 2),
                          config.get("max_gpus_per_pr", 4), config.get("max_gpus", 8))


def make_backend(config: dict[str, Any], backend: str, directory: Path) -> Any:
    from .backends import KubernetesBackend, ModalBackend
    cls = {"vllm": KubernetesBackend, "modal": ModalBackend}.get(backend)
    if cls is None:
        raise ValueError("The new dispatcher supports only modal and vllm")
    backend_config = config["kubernetes" if backend == "vllm" else "modal"]
    return cls(backend_config, Path(__file__).resolve().parent, directory)


def _run_lane(backend: Any, handle: dict[str, Any], request: dict[str, Any], lane: dict[str, Any],
              cancel: threading.Event) -> tuple[int, bool, str]:
    error = ""
    try:
        code = backend.run(handle, request, lane, cancel)
        if type(code) is not int or not 0 <= code <= 255:
            raise RuntimeError("Backend did not return a valid process exit status")
    except Exception as exception:
        code, error = INFRASTRUCTURE_FAILURE, str(exception)
    try:
        stopped = backend.stop(handle)
    except Exception as exception:
        stopped = False
        error += f"; stop failed: {exception}"
    if not stopped:
        code = INFRASTRUCTURE_FAILURE
        error += "; worker termination unconfirmed; reservation retained"
    return code, stopped, error


def execute(request: dict[str, Any], config: dict[str, Any], *, backend: Any = None,
            cancel: threading.Event | None = None, poll_seconds: float = 2.0) -> int:
    """Run one suite; independent processes share the PR/GPU admission ledger."""
    cancel = cancel or threading.Event()
    build_id = request["build_id"]
    directory = run_directory(config, build_id)
    store = make_store(config)
    if any(lane["gpus"] > min(store.max_gpus_per_pr, store.max_gpus) for lane in request["lanes"]):
        raise ValueError("Selected lane does not fit the configured GPU budget")
    backend = backend or make_backend(config, request["backend"], directory)
    results: dict[str, Any] = {}
    code = 0
    with build_lock(directory):
        # Never reattach to an ambiguous prior launch. Buildkite retries have
        # a new job ID; abandoned attempts need explicit backend reconciliation.
        if any(item["build_id"] == build_id for item in store.snapshot()["builds"]):
            raise RuntimeError("Build attempt already exists; inspect/recover it instead of replaying it")
        store.register(build_id, request["pr_key"], request["backend"], request["commit"])
        (directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        deadline = time.monotonic() + config.get("queue_timeout_seconds", 21600)
        # A docs-only merge plan requires no GPUs and must not wait behind
        # unrelated PRs. Arrival order is not evidence of commit recency, so
        # never cancel other SHAs merely because this request arrived later.
        admitted = not request["lanes"]
        while not admitted:
            store.heartbeat(build_id)
            if cancel.is_set() or store.is_cancelled(build_id) or time.monotonic() >= deadline:
                cancel.set()
                store.request_cancel(build_id)
                store.finish(build_id)
                code = INFRASTRUCTURE_FAILURE
                break
            admitted = store.try_admit(build_id)
            if not admitted:
                cancel.wait(poll_seconds)
        pending = list(request["lanes"]) if admitted else []
        golden_selected = any(lane["key"] == "golden-gate" for lane in pending)
        running: dict[Future[tuple[int, bool, str]], dict[str, Any]] = {}
        # Lane concurrency is additionally bounded by transactional GPU leases.
        with ThreadPoolExecutor(max_workers=4) as pool:
            while pending or running:
                store.heartbeat(build_id)
                if store.is_cancelled(build_id):
                    cancel.set()
                if cancel.is_set():
                    store.request_cancel(build_id)
                    for lane in pending:
                        results[lane["key"]] = {"state": "cancelled"}
                    pending.clear()
                    code = code or INFRASTRUCTURE_FAILURE
                for future, lane in list(running.items()):
                    if not future.done():
                        continue
                    lane_code, stopped, error = future.result()
                    if stopped:
                        store.release(build_id, lane["key"])
                    else:
                        cancel.set()
                    results[lane["key"]] = {"state": "passed" if lane_code == 0 else "failed",
                                             "exit_code": lane_code, "stopped": stopped, "error": error}
                    print(f"--- {lane['key']}: exit {lane_code}", flush=True)
                    code = code or lane_code
                    del running[future]
                for lane in list(pending):
                    if cancel.is_set() or len(running) >= 4:
                        break
                    if golden_selected and not lane["fastcheck"] and lane["key"] != "golden-gate":
                        golden = results.get("golden-gate")
                        if golden is None:
                            continue
                        if golden["state"] != "passed":
                            results[lane["key"]] = {"state": "skipped", "reason": "golden gate failed"}
                            pending.remove(lane)
                            continue
                    handle = backend.handle(build_id, lane["key"])
                    if store.try_acquire(build_id, lane["key"], lane["gpus"], handle):
                        print(f"--- Starting {lane['key']} on {request['backend']} ({lane['gpus']} GPUs)", flush=True)
                        future = pool.submit(_run_lane, backend, handle, request, lane, cancel)
                        running[future] = lane
                        pending.remove(lane)
                if pending or running:
                    # wait() on an already-set Event spins; cancellation still
                    # polls bounded backend shutdowns at the normal interval.
                    time.sleep(poll_seconds)
        active = [a for a in store.snapshot()["allocations"] if a["build_id"] == build_id and a["state"] == "active"]
        if not active:
            store.finish(build_id)
        else:
            code = INFRASTRUCTURE_FAILURE
        summary = {"build_id": build_id, "backend": request["backend"], "commit": request["commit"],
                   "exit_code": code, "recovery_required": bool(active), "lanes": results}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    return code


def recover(config: dict[str, Any], build_id: str, *, backend: Any = None) -> None:
    """Stop abandoned workers first, then release reservations. Never TTL-unlock."""
    store = make_store(config)
    build = next((b for b in store.snapshot()["builds"] if b["build_id"] == build_id), None)
    if build is None:
        raise ValueError("Unknown build ID")
    directory = run_directory(config, build_id)
    with build_lock(directory):
        store.request_cancel(build_id)
        backend = backend or make_backend(config, build["backend"], directory)
        uncertain = []
        for allocation in store.snapshot()["allocations"]:
            if allocation["build_id"] != build_id or allocation["state"] != "active":
                continue
            try:
                stopped = backend.stop(allocation["handle"])
            except Exception:
                stopped = False
            if stopped:
                store.release(build_id, allocation["lane_id"])
            else:
                uncertain.append(allocation["lane_id"])
        if uncertain:
            raise RuntimeError(f"Could not confirm termination; reservations retained: {uncertain}")
        store.finish(build_id)


def upload_artifacts(directory: Path) -> None:
    """Only upload controller-owned logs/metadata, never arbitrary PR globs."""
    for path in sorted(directory.iterdir()):
        if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".log"):
            subprocess.run(["buildkite-agent", "artifact", "upload", path.name], cwd=directory, check=True)


@contextmanager
def cancellation_signals(event: threading.Event) -> Iterator[None]:
    previous: dict[signal.Signals, Any] = {}
    def handler(signum: int, frame: Any) -> None:
        event.set()
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, old_handler in previous.items():
            signal.signal(signum, old_handler)
