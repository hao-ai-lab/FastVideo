import multiprocessing as mp
from multiprocessing import resource_tracker, util
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from fastvideo.fastvideo_args import FastVideoArgs
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


class _FailingWorker:

    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1
        raise RuntimeError("interrupted shutdown")


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


def test_worker_shutdown_still_idempotent_after_failure(monkeypatch) -> None:
    worker = _FailingWorker()
    compile_shutdowns = []
    proc = WorkerMultiprocProc.__new__(WorkerMultiprocProc)
    proc.rank = 0
    proc.worker = worker
    proc._shutdown_started = False
    proc._shutdown_complete = False
    proc._shutdown_response = None
    monkeypatch.setattr(
        "fastvideo.worker.multiproc_executor._shutdown_torch_compile_workers",
        lambda: compile_shutdowns.append(True),
    )

    first = proc.shutdown()
    second = proc.shutdown()

    assert first == {"status": "shutdown"}
    assert second == {"status": "shutdown"}
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


def test_compile_worker_cleanup_dispatch_target_exists_in_real_inductor() -> None:
    """Bind the cleanup dispatch to the real torch API, not a stub.

    ``_shutdown_torch_compile_workers`` looks up
    ``sys.modules["torch._inductor.async_compile"].shutdown_compile_workers``.
    The tests above stub that object, so they cannot notice a torch release that
    renames or moves the private API: the dispatch would silently become a
    no-op and the orphaned compile-subprocess regression would return while the
    suite stays green. Fail loudly on such an upgrade instead.
    """
    async_compile = pytest.importorskip(
        "torch._inductor.async_compile",
        reason="the dispatch contract only matters where torch ships Inductor",
    )

    assert sys.modules.get("torch._inductor.async_compile") is async_compile
    assert callable(getattr(async_compile, "shutdown_compile_workers", None))


def _fake_worker_handle(**kwargs):
    rank = kwargs["rank"]
    proc = SimpleNamespace(is_alive=lambda: False, exitcode=0, pid=10_000 + rank)
    return SimpleNamespace(proc=proc, rank=rank, pipe=SimpleNamespace(), reader=SimpleNamespace())


@pytest.fixture
def captured_worker_kwargs(monkeypatch) -> list[dict]:
    """Stub worker spawn so MultiprocExecutor.__init__ only records worker kwargs."""
    captured: list[dict] = []

    def make_worker_process(**kwargs):
        captured.append(kwargs)
        return _fake_worker_handle(**kwargs)

    monkeypatch.setattr(WorkerMultiprocProc, "make_worker_process", staticmethod(make_worker_process))
    monkeypatch.setattr(WorkerMultiprocProc, "wait_for_ready", staticmethod(lambda handles: handles))
    monkeypatch.setattr("fastvideo.worker.multiproc_executor.set_multiproc_executor_envs", lambda: None)
    monkeypatch.setattr("fastvideo.worker.multiproc_executor.get_open_port", lambda _port=None: 29500)
    monkeypatch.setattr("fastvideo.worker.multiproc_executor.get_loopback_ip", lambda: "127.0.0.1")
    monkeypatch.setattr("fastvideo.worker.multiproc_executor.atexit.register", lambda *_args, **_kwargs: None)
    return captured


@pytest.fixture
def spawn_probe(monkeypatch):
    # A bare module avoids importing the entire FastVideo package in the CPU child.
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from spawn_ipc_probe import probe
    return probe


def _close_queues(executor: MultiprocExecutor) -> None:
    for queue in (executor._streaming_input_queue, executor._streaming_output_queue):
        if queue is not None:
            queue.close()
            queue.join_thread()


def _stop(process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def test_streaming_ipc_queues_follow_fastvideo_args(captured_worker_kwargs) -> None:
    disabled = MultiprocExecutor(FastVideoArgs(model_path="test/model", num_gpus=2))
    assert disabled._streaming_input_queue is None
    assert disabled._streaming_output_queue is None
    assert all(call["streaming_input_queue"] is None for call in captured_worker_kwargs)
    assert all(call["streaming_output_queue"] is None for call in captured_worker_kwargs)

    captured_worker_kwargs.clear()
    enabled = MultiprocExecutor(
        FastVideoArgs(model_path="test/model", num_gpus=2, enable_streaming_ipc_queues=True))
    assert enabled._streaming_input_queue is not None
    assert enabled._streaming_output_queue is not None
    assert all(call["streaming_input_queue"] is not None for call in captured_worker_kwargs)
    assert all(call["streaming_output_queue"] is not None for call in captured_worker_kwargs)
    _close_queues(enabled)


def test_enable_streaming_requires_ipc_queues() -> None:
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor._streaming_enabled = False
    executor._streaming_input_queue = None
    executor._streaming_output_queue = None
    with pytest.raises(RuntimeError, match="Streaming IPC queues are not initialized"):
        executor.enable_streaming()


@pytest.mark.parametrize("enabled", [False, True])
def test_streaming_queues_survive_real_spawn(captured_worker_kwargs, spawn_probe, enabled) -> None:
    executor = MultiprocExecutor(
        FastVideoArgs(model_path="test/model", num_gpus=2, enable_streaming_ipc_queues=enabled))
    context = mp.get_context("spawn")
    try:
        for call in captured_worker_kwargs:
            receive, send = context.Pipe(duplex=False)
            process = context.Process(target=spawn_probe,
                                      args=(call["streaming_input_queue"], call["streaming_output_queue"], send))
            try:
                process.start()
                send.close()
                if enabled:
                    executor._streaming_input_queue.put(41)
                assert receive.poll(60), "Spawn child did not rebuild IPC and respond"
                assert receive.recv() == ("enabled" if enabled else "disabled")
                if enabled:
                    assert executor._streaming_output_queue.get(timeout=10) == 42
                process.join(timeout=10)
                assert process.exitcode == 0
            finally:
                _stop(process)
                receive.close()
                send.close()
    finally:
        executor.shutting_down = True  # Worker handles are fixtures, not live executor workers.
        _close_queues(executor)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="inspects POSIX semaphores under /dev/shm")
def test_standard_workers_survive_semaphore_removal_during_spawn(captured_worker_kwargs, spawn_probe) -> None:
    """Regression for SemLock._rebuild ENOENT: standard workers must not unpickle streaming semaphores.

    Workers are held at an import barrier while the parent unlinks a live semaphore
    from /dev/shm, as a Slurm epilog cleaning the user's /dev/shm does. Without streaming queues, the workers
    have nothing to rebuild and must still start and reply.
    """
    executor = MultiprocExecutor(FastVideoArgs(model_path="test/model", num_gpus=2))
    executor.shutting_down = True  # Fake executor handles; real test children are owned below.
    assert len(captured_worker_kwargs) == 2
    context = mp.get_context("spawn")
    sentinel = context.Lock()  # Not passed to children; proves deletion really happens.
    name = sentinel._semlock.name
    path = "/dev/shm/sem." + name.lstrip("/")
    from spawn_ipc_probe import gate_dir  # importable once spawn_probe has extended sys.path

    gate = gate_dir(os.getpid())
    # A stale gate left behind by a killed run would silently release the barrier.
    shutil.rmtree(gate, ignore_errors=True)
    gate.mkdir()
    children, pipes = [], []
    try:
        for call in captured_worker_kwargs:
            receive, send = context.Pipe(duplex=False)
            pipes.append((receive, send))
            child = context.Process(target=spawn_probe,
                                    args=(call["streaming_input_queue"], call["streaming_output_queue"], send))
            children.append(child)
            child.start()
            send.close()
        deadline = time.monotonic() + 60
        while not all((gate / f"{child.pid}.ready").exists() for child in children):
            assert all(child.exitcode is None for child in children), "Worker exited before import barrier"
            assert time.monotonic() < deadline, "Worker did not reach import barrier"
            time.sleep(.01)

        os.unlink(path)  # Exact name created above; never scan or delete by pattern.
        resource_tracker.unregister(name, "semaphore")
        for finalizer in list(util._finalizer_registry.values()):
            if getattr(finalizer, "_args", ()) == (name,):
                finalizer.cancel()
        assert not os.path.exists(path)
        (gate / "release").touch()

        for child, (receive, _) in zip(children, pipes):
            assert receive.poll(30), "Worker failed to start after semaphore removal"
            assert receive.recv() == "disabled"
            child.join(timeout=30)
            assert child.exitcode == 0
            assert not (gate / f"{child.pid}.stderr").read_text()
    finally:
        for child in children:
            if child.pid:
                _stop(child)
        for receive, send in pipes:
            receive.close()
            send.close()
        shutil.rmtree(gate, ignore_errors=True)


@pytest.mark.parametrize(("use_queue_mode", "expected"), [(True, True), (False, False)])
def test_streaming_generator_enables_queues_only_for_queue_mode(monkeypatch, use_queue_mode, expected) -> None:
    from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
    from fastvideo.entrypoints.video_generator import VideoGenerator

    seen: list[FastVideoArgs] = []

    def fake_init(self, fastvideo_args, executor_class, log_stats, **kwargs):
        seen.append(fastvideo_args)
        self.executor = MultiprocExecutor.__new__(MultiprocExecutor)

    monkeypatch.setattr(VideoGenerator, "__init__", fake_init)
    args = FastVideoArgs(model_path="test/model")

    StreamingVideoGenerator(args, MultiprocExecutor, log_stats=False, use_queue_mode=use_queue_mode)

    assert seen[0].enable_streaming_ipc_queues is expected
    assert args.enable_streaming_ipc_queues is False  # the caller's args are never mutated


def test_streaming_generator_from_fastvideo_args_forwards_log_queue(monkeypatch) -> None:
    from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
    from fastvideo.entrypoints.video_generator import VideoGenerator
    from fastvideo.worker.executor import Executor

    seen: list[tuple[FastVideoArgs, object]] = []

    def fake_init(self, fastvideo_args, executor_class, log_stats, **kwargs):
        seen.append((fastvideo_args, kwargs.get("log_queue")))
        self.executor = MultiprocExecutor.__new__(MultiprocExecutor)

    monkeypatch.setattr(VideoGenerator, "__init__", fake_init)
    monkeypatch.setattr(Executor, "get_class", staticmethod(lambda fastvideo_args: MultiprocExecutor))
    args = FastVideoArgs(model_path="test/model")
    sentinel = object()

    StreamingVideoGenerator.from_fastvideo_args(args, log_queue=sentinel)

    assert seen[0][1] is sentinel  # VideoGenerator.from_config's log_queue= call site works again
    assert seen[0][0] is not args and seen[0][0].enable_streaming_ipc_queues is True
    assert args.enable_streaming_ipc_queues is False
