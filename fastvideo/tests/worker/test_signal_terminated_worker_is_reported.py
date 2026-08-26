# SPDX-License-Identifier: Apache-2.0
"""A worker killed by a signal has to say which signal it received.

`worker_main` installs a SIGTERM/SIGINT handler that raises `SystemExit`. That is
a `BaseException`, not an `Exception`, so it walks straight past a bare
`except Exception` and the worker exits having reported nothing. The parent then
sees a closed pipe and raises "See stack trace for root cause" with no stack
trace, which is indistinguishable from a hang, a crash, or an out-of-memory kill.

That matters on machines that terminate processes under memory pressure. A DGX
Spark ships earlyoom configured to send SIGTERM to python first once available
memory drops below six percent, so this is the common failure there and it is
completely silent.

The tests replace startup and worker construction with lightweight signal
injection, so they exercise the real handler and outer exception path without
initializing a distributed process group or model.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys

import pytest

import fastvideo.worker.multiproc_executor as multiproc_executor
from fastvideo.worker.multiproc_executor import WorkerMultiprocProc

# The spawn context reimports PyTorch and FastVideo in every child. That takes
# nearly five seconds even on the local GB10 and can take longer on a loaded ARM
# CI node, so a five-second startup deadline makes the signal contract flaky.
_SUBPROCESS_TIMEOUT_SECONDS = 30


class _ReadyPipe:

    def __init__(self) -> None:
        self.closed = False
        self.messages = []

    def send(self, message) -> None:
        self.messages.append(message)

    def close(self) -> None:
        self.closed = True


class _ParentlessProcess:

    def parent(self):
        return None


def _run_real_signal_probe(ready_pipe, log_queue, signal_point, entered_signal_point, shutdown_called) -> None:

    class SignalWaitingWorker:

        READY_STR = "READY"

        def __init__(self, *args, **kwargs) -> None:
            if signal_point == "worker_construction":
                entered_signal_point.set()
                signal.pause()

        def worker_busy_loop(self) -> None:
            assert signal_point == "worker_loop"
            entered_signal_point.set()
            signal.pause()

        def shutdown(self) -> None:
            shutdown_called.set()

    multiproc_executor.kill_itself_when_parent_died = lambda: None
    multiproc_executor.faulthandler.enable = lambda: None
    multiproc_executor.psutil.Process = _ParentlessProcess
    multiproc_executor.WorkerMultiprocProc = SignalWaitingWorker
    WorkerMultiprocProc.worker_main(ready_pipe=ready_pipe, rank=7, log_queue=log_queue)


@pytest.mark.parametrize(
    ("signum", "expected_reason", "unexpected_reason"),
    [
        (signal.SIGTERM, "parent cleaning up workers", "user interrupted"),
        (signal.SIGINT, "user interrupted", "out-of-memory daemon"),
    ],
)
@pytest.mark.parametrize("injection_point", ["startup", "worker_construction", "worker_loop"])
def test_worker_main_reports_received_signal(monkeypatch: pytest.MonkeyPatch, signum: int, expected_reason: str,
                                             unexpected_reason: str, injection_point: str) -> None:
    installed_handlers = {}
    logged_messages = []
    workers = []
    ready_pipe = _ReadyPipe()

    def install_handler(installed_signum, handler):
        installed_handlers[installed_signum] = handler

    def inject_signal() -> None:
        installed_handlers[signum](signum, None)

    class SignalledWorker:

        READY_STR = "READY"

        def __init__(self, *args, **kwargs) -> None:
            self.shutdown_called = False
            workers.append(self)
            if injection_point == "worker_construction":
                inject_signal()

        def worker_busy_loop(self) -> None:
            assert injection_point == "worker_loop"
            inject_signal()

        def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setattr(multiproc_executor.signal, "signal", install_handler)
    monkeypatch.setattr(multiproc_executor, "kill_itself_when_parent_died",
                        inject_signal if injection_point == "startup" else lambda: None)
    monkeypatch.setattr(multiproc_executor.faulthandler, "enable", lambda: None)
    monkeypatch.setattr(multiproc_executor.psutil, "Process", _ParentlessProcess)
    monkeypatch.setattr(multiproc_executor, "WorkerMultiprocProc", SignalledWorker)
    monkeypatch.setattr(multiproc_executor.logger, "exception",
                        lambda message, *args: logged_messages.append(message % args))

    with pytest.raises(SystemExit) as exc_info:
        WorkerMultiprocProc.worker_main(ready_pipe=ready_pipe, rank=7)

    assert exc_info.value.code is None
    assert exc_info.value.signum == signum
    assert ready_pipe.closed
    assert len(logged_messages) == 1
    assert f"Worker 7 received {signal.Signals(signum).name} ({signum})" in logged_messages[0]
    assert expected_reason in logged_messages[0]
    assert unexpected_reason not in logged_messages[0]
    if injection_point == "worker_loop":
        assert ready_pipe.messages == [{"status": "READY"}]
        assert workers[0].shutdown_called


def test_worker_main_preserves_unrelated_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    logged_messages = []
    ready_pipe = _ReadyPipe()

    class ExitingWorker:

        def __init__(self, *args, **kwargs) -> None:
            raise SystemExit(23)

    monkeypatch.setattr(multiproc_executor.signal, "signal", lambda *args: None)
    monkeypatch.setattr(multiproc_executor, "kill_itself_when_parent_died", lambda: None)
    monkeypatch.setattr(multiproc_executor.faulthandler, "enable", lambda: None)
    monkeypatch.setattr(multiproc_executor.psutil, "Process", _ParentlessProcess)
    monkeypatch.setattr(multiproc_executor, "WorkerMultiprocProc", ExitingWorker)
    monkeypatch.setattr(multiproc_executor.logger, "exception",
                        lambda message, *args: logged_messages.append(message % args))

    with pytest.raises(SystemExit) as exc_info:
        WorkerMultiprocProc.worker_main(ready_pipe=ready_pipe, rank=7)

    assert exc_info.value.code == 23
    assert ready_pipe.closed
    assert logged_messages == []


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX signal delivery is a Linux worker contract")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("signal_point", ["worker_construction", "worker_loop"])
def test_worker_main_forwards_real_signal_traceback_across_processes(signum: int, signal_point: str) -> None:
    context = mp.get_context("spawn")
    entered_signal_point = context.Event()
    shutdown_called = context.Event()
    parent_ready_pipe, child_ready_pipe = context.Pipe(duplex=False)
    log_queue = context.Queue()

    process = context.Process(
        target=_run_real_signal_probe,
        args=(child_ready_pipe, log_queue, signal_point, entered_signal_point, shutdown_called),
    )
    process.start()
    child_ready_pipe.close()

    try:
        reached_signal_point = entered_signal_point.wait(timeout=_SUBPROCESS_TIMEOUT_SECONDS)
        assert reached_signal_point, "child did not reach the requested signal point"
        if signal_point == "worker_loop":
            assert parent_ready_pipe.recv() == {"status": "READY"}

        os.kill(process.pid, signum)
        process.join(timeout=_SUBPROCESS_TIMEOUT_SECONDS)
        assert not process.is_alive(), "signalled worker did not exit"
        assert process.exitcode == 0  # Preserve the historical argument-less SystemExit status.

        record = log_queue.get(timeout=_SUBPROCESS_TIMEOUT_SECONDS)
        message = record.getMessage()
        assert f"Worker 7 received {signal.Signals(signum).name} ({signum})" in message
        assert "Traceback (most recent call last)" in message
        assert "The stack below is where execution was interrupted, not the cause." in message

        if signal_point == "worker_loop":
            assert shutdown_called.wait(timeout=1)
        else:
            assert not shutdown_called.is_set()
            with pytest.raises(EOFError):
                parent_ready_pipe.recv()
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=_SUBPROCESS_TIMEOUT_SECONDS)
        parent_ready_pipe.close()
        log_queue.close()
        log_queue.join_thread()
