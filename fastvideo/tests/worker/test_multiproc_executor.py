import sys
from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.worker.multiproc_executor import (
    _RPC_ERROR_KEY,
    _WORKER_GRACEFUL_SHUTDOWN_TIMEOUT_S,
    _raise_for_rpc_errors,
    _shutdown_torch_compile_workers,
    MultiprocExecutor,
    WorkerMultiprocProc,
    WorkerProcHandle,
)


class _ScriptedPipe:

    def __init__(self, messages):
        self.messages = list(messages)
        self.responses = []

    def recv(self):
        return self.messages.pop(0)

    def send(self, response):
        self.responses.append(response)


class _RecoveringWorker:

    def __init__(self):
        self.calls = 0

    def execute_forward(self, forward_batch, fastvideo_args):
        del forward_batch, fastvideo_args
        self.calls += 1
        if self.calls == 1:
            raise ValueError("bad request")
        return ForwardBatch(data_type="video", output=torch.ones(1))

    def shutdown(self):
        return {"status": "shutdown"}


class _CountingWorker:

    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1
        return {"status": "shutdown"}


class _GracefulProcess:

    def __init__(self, required_timeout: float):
        self.required_timeout = required_timeout
        self.alive = True
        self.join_timeouts = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if timeout is not None and timeout >= self.required_timeout:
            self.alive = False

    def terminate(self):
        self.terminate_calls += 1
        self.alive = False

    def kill(self):
        self.kill_calls += 1
        self.alive = False


class _StuckProcess(_GracefulProcess):

    def terminate(self):
        self.terminate_calls += 1


class _RecordingPipe:

    def __init__(self):
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


def test_worker_rpc_error_does_not_exit_busy_loop(monkeypatch) -> None:
    request = {
        "method": "execute_forward",
        "kwargs": {
            "forward_batch": SimpleNamespace(),
            "fastvideo_args": SimpleNamespace(),
        },
    }
    pipe = _ScriptedPipe([request, request, {"method": "shutdown"}])
    proc = WorkerMultiprocProc.__new__(WorkerMultiprocProc)
    proc.rank = 0
    proc.pipe = pipe
    proc.worker = _RecoveringWorker()
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    proc.worker_busy_loop()

    assert pipe.responses[0][_RPC_ERROR_KEY] is True
    assert "ValueError: bad request" in pipe.responses[0]["error"]
    assert torch.equal(pipe.responses[1]["output_batch"], torch.ones(1))
    assert pipe.responses[2] == {"status": "shutdown"}


def test_parent_raises_worker_rpc_error() -> None:
    with pytest.raises(RuntimeError, match="worker 0: ValueError: bad request"):
        _raise_for_rpc_errors("execute_forward", [{_RPC_ERROR_KEY: True, "error": "ValueError: bad request"}])


def test_worker_shutdown_closes_worker_and_compile_pool_once(monkeypatch) -> None:
    worker = _CountingWorker()
    compile_shutdowns = []
    proc = WorkerMultiprocProc.__new__(WorkerMultiprocProc)
    proc.worker = worker
    proc._shutdown_complete = False
    proc._shutdown_response = None
    monkeypatch.setattr(
        "fastvideo.worker.multiproc_executor._shutdown_torch_compile_workers",
        lambda: compile_shutdowns.append(True),
    )

    first = proc.shutdown()
    second = proc.shutdown()

    assert first == {"status": "shutdown"}
    assert second == first
    assert worker.shutdown_calls == 1
    assert compile_shutdowns == [True]


def test_compile_worker_cleanup_uses_only_an_already_loaded_inductor(monkeypatch) -> None:
    compile_shutdowns = []
    fake_async_compile = SimpleNamespace(shutdown_compile_workers=lambda: compile_shutdowns.append(True))
    monkeypatch.setitem(sys.modules, "torch._inductor.async_compile", fake_async_compile)

    _shutdown_torch_compile_workers()

    assert compile_shutdowns == [True]


def test_compile_worker_cleanup_does_not_import_inductor(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "torch._inductor.async_compile", raising=False)

    _shutdown_torch_compile_workers()

    assert "torch._inductor.async_compile" not in sys.modules


def test_executor_allows_slow_graceful_worker_exit_before_sigterm() -> None:
    # Regression for torch.compile workers: Inductor cleanup can take longer
    # than the old five-second grace period, especially under `nice -n 19`.
    process = _GracefulProcess(required_timeout=6.0)
    pipe = _RecordingPipe()
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.shutting_down = False
    executor.workers = [WorkerProcHandle(proc=process, rank=0, pipe=pipe)]

    executor.shutdown()

    assert pipe.messages == [{"method": "shutdown", "args": (), "kwargs": {}}]
    assert pipe.closed is True
    assert process.join_timeouts[0] > 6.0
    assert process.join_timeouts[0] <= _WORKER_GRACEFUL_SHUTDOWN_TIMEOUT_S
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert executor.workers == []


def test_executor_still_kills_worker_that_ignores_graceful_exit_and_sigterm() -> None:
    process = _StuckProcess(required_timeout=float("inf"))
    pipe = _RecordingPipe()
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.shutting_down = False
    executor.workers = [WorkerProcHandle(proc=process, rank=0, pipe=pipe)]

    executor.shutdown()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.alive is False
    assert pipe.closed is True
    assert executor.workers == []
