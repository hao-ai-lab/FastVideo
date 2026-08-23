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

The test replaces worker construction with a lightweight signal injection, so
it exercises the real handler and outer exception path without initializing a
distributed process group or model.
"""
from __future__ import annotations

import signal

import pytest

import fastvideo.worker.multiproc_executor as multiproc_executor
from fastvideo.worker.multiproc_executor import WorkerMultiprocProc


class _ReadyPipe:

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ParentlessProcess:

    def parent(self):
        return None


@pytest.mark.parametrize(
    ("signum", "expected_reason", "unexpected_reason"),
    [
        (signal.SIGTERM, "out-of-memory daemon", "user interrupted"),
        (signal.SIGINT, "user interrupted", "out-of-memory daemon"),
    ],
)
def test_worker_main_reports_received_signal(monkeypatch: pytest.MonkeyPatch, signum: int, expected_reason: str,
                                             unexpected_reason: str) -> None:
    installed_handlers = {}
    logged_messages = []
    ready_pipe = _ReadyPipe()

    def install_handler(installed_signum, handler):
        installed_handlers[installed_signum] = handler

    class SignalledWorker:

        def __init__(self, *args, **kwargs) -> None:
            installed_handlers[signum](signum, None)

    monkeypatch.setattr(multiproc_executor.signal, "signal", install_handler)
    monkeypatch.setattr(multiproc_executor, "kill_itself_when_parent_died", lambda: None)
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
