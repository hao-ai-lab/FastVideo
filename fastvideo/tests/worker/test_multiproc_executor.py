from collections import OrderedDict, namedtuple
import multiprocessing as mp
from threading import Thread
from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.worker.multiproc_executor import (
    _RPC_ERROR_KEY,
    _copy_mps_tensors_to_cpu,
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


_Audio = namedtuple("_Audio", ["samples", "sample_rate"])


class _OutputWorker:

    def __init__(self, output_batch):
        self.output_batch = output_batch

    def execute_forward(self, forward_batch, fastvideo_args):
        return self.output_batch

    def shutdown(self):
        return {"status": "shutdown"}


def _execute_forward_over_pipe(output_batch):
    """Exercise the real response path and PyTorch's multiprocessing reducers."""
    parent_pipe, worker_pipe = mp.Pipe()
    proc = WorkerMultiprocProc.__new__(WorkerMultiprocProc)
    proc.rank = 0
    proc.pipe = worker_pipe
    proc.worker = _OutputWorker(output_batch)
    thread = Thread(target=proc.worker_busy_loop, daemon=True)
    thread.start()
    try:
        parent_pipe.send({
            "method": "execute_forward",
            "kwargs": {
                "forward_batch": SimpleNamespace(),
                "fastvideo_args": SimpleNamespace(),
            },
        })
        assert parent_pipe.poll(10), "Worker did not return a response"
        response = parent_pipe.recv()
        parent_pipe.send({"method": "shutdown"})
        assert parent_pipe.poll(10), "Worker did not shut down"
        assert parent_pipe.recv() == {"status": "shutdown"}
        thread.join(timeout=10)
        assert not thread.is_alive()
        return response
    finally:
        parent_pipe.close()
        worker_pipe.close()


@pytest.mark.parametrize("device, tensor_field", [
    ("cpu", "all"),
    ("mps", "output"),
    ("mps", "extra"),
    ("mps", "trajectory_latents"),
    ("mps", "trajectory_timesteps"),
    ("mps", "logging_info"),
    ("mps", "all"),
])
def test_execute_forward_response_serializes_device_tensors(monkeypatch, device, tensor_field):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Requires an Apple GPU with MPS")

    def tensor(field, dtype=torch.float32):
        target = device if tensor_field in (field, "all") else "cpu"
        return torch.arange(6, dtype=dtype, device=target).reshape(2, 3)

    output_batch = ForwardBatch(
        data_type="video",
        output=tensor("output"),
        extra={
            "audio": [_Audio(tensor("extra"), 24000)],
            "metadata": OrderedDict(ids=tensor("extra", torch.int32), label="test", optional=None),
        },
        trajectory_latents=tensor("trajectory_latents", torch.float16),
        trajectory_timesteps=[tensor("trajectory_timesteps", torch.int32)],
    )
    output_batch.logging_info.add_stage_metric("decode", "metric", {"values": (tensor("logging_info"),)})
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr("fastvideo.worker.multiproc_executor.envs.FASTVIDEO_STAGE_LOGGING", True)

    response = _execute_forward_over_pipe(output_batch)

    assert _RPC_ERROR_KEY not in response, response.get("error")

    def assert_tensor(actual, expected):
        assert actual.device.type == "cpu"
        assert not actual.requires_grad
        torch.testing.assert_close(actual, expected.detach().cpu(), rtol=0, atol=0)

    assert_tensor(response["output_batch"], output_batch.output)
    assert isinstance(response["extra"]["audio"], list)
    assert type(response["extra"]["audio"][0]) is _Audio
    assert_tensor(response["extra"]["audio"][0].samples, output_batch.extra["audio"][0].samples)
    assert response["extra"]["audio"][0].sample_rate == 24000
    assert type(response["extra"]["metadata"]) is OrderedDict
    assert_tensor(response["extra"]["metadata"]["ids"], output_batch.extra["metadata"]["ids"])
    assert response["extra"]["metadata"]["label"] == "test"
    assert response["extra"]["metadata"]["optional"] is None
    assert response["extra"]["peak_memory_mb"] == 0
    assert_tensor(response["trajectory_latents"], output_batch.trajectory_latents)
    assert_tensor(response["trajectory_timesteps"][0], output_batch.trajectory_timesteps[0])
    assert type(response["logging_info"]) is type(output_batch.logging_info)
    assert response["logging_info"].get_execution_order() == ["decode"]
    actual_metric = response["logging_info"].get_stage_info("decode")["metric"]["values"]
    expected_metric = output_batch.logging_info.get_stage_info("decode")["metric"]["values"]
    assert type(actual_metric) is tuple
    assert_tensor(actual_metric[0], expected_metric[0])
    # Preparing the response must not offload the worker's original tensors.
    original_tensors = {
        "output": output_batch.output,
        "extra": output_batch.extra["audio"][0].samples,
        "trajectory_latents": output_batch.trajectory_latents,
        "trajectory_timesteps": output_batch.trajectory_timesteps[0],
        "logging_info": expected_metric[0],
    }
    for field, original in original_tensors.items():
        expected_device = device if tensor_field in (field, "all") else "cpu"
        assert original.device.type == expected_device


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_response_preparation_preserves_non_mps_tensor_identity(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires an NVIDIA GPU with CUDA")
    tensor = torch.ones(2, 3, device=device, requires_grad=True) * 2
    response = {"output_batch": tensor, "extra": {"nested": [tensor, (tensor,)]}}

    prepared = _copy_mps_tensors_to_cpu(response)

    assert prepared["output_batch"] is tensor
    assert prepared["extra"]["nested"][0] is tensor
    assert prepared["extra"]["nested"][1][0] is tensor
    assert tensor.requires_grad


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires an Apple GPU with MPS")
def test_execute_forward_detaches_mps_result_before_serializing(monkeypatch):
    tensor = torch.arange(6, device="mps", dtype=torch.float32).requires_grad_() * 2
    output_batch = ForwardBatch(data_type="video", output=tensor)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    response = _execute_forward_over_pipe(output_batch)

    assert _RPC_ERROR_KEY not in response, response.get("error")
    assert response["output_batch"].device.type == "cpu"
    assert not response["output_batch"].requires_grad
    torch.testing.assert_close(response["output_batch"], tensor.detach().cpu(), rtol=0, atol=0)
    assert tensor.device.type == "mps"
    assert tensor.requires_grad
