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
    _raise_for_rpc_errors,
    MultiprocExecutor,
    WorkerMultiprocProc,
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
