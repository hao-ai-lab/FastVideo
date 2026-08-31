from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.worker.multiproc_executor import (
    _RPC_ERROR_KEY,
    _raise_for_rpc_errors,
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
