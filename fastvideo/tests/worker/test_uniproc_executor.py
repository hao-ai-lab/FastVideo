# SPDX-License-Identifier: Apache-2.0
from inspect import signature
from types import SimpleNamespace

import pytest

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ForwardBatch
from fastvideo.worker.executor import Executor
from fastvideo.worker.multiproc_executor import MultiprocExecutor
from fastvideo.worker.uniproc_executor import UniprocExecutor


def test_uniproc_executor_implements_executor_abc() -> None:
    remaining = getattr(UniprocExecutor, "__abstractmethods__", frozenset())
    assert remaining == frozenset(), remaining


def test_get_class_uses_uniproc_for_single_gpu_mp() -> None:
    args = FastVideoArgs(model_path="test", num_gpus=1, distributed_executor_backend="mp")
    assert Executor.get_class(args) is UniprocExecutor


def test_get_class_uses_uniproc_when_backend_is_uni() -> None:
    args = FastVideoArgs(model_path="test", num_gpus=1, distributed_executor_backend="uni")
    assert Executor.get_class(args) is UniprocExecutor


def test_get_class_uses_multiproc_for_multi_gpu_mp() -> None:
    args = FastVideoArgs(model_path="test", num_gpus=2, distributed_executor_backend="mp")
    assert Executor.get_class(args) is MultiprocExecutor


def test_uniproc_rejects_multi_gpu() -> None:
    args = FastVideoArgs(model_path="test", num_gpus=2, distributed_executor_backend="uni")
    with pytest.raises(ValueError, match="only supports num_gpus=1"):
        UniprocExecutor(args)


def test_uniproc_init_does_not_use_mp_context(monkeypatch) -> None:

    def boom() -> None:
        raise AssertionError("UniprocExecutor must not spawn a multiprocessing context")

    monkeypatch.setattr("fastvideo.utils.get_mp_context", boom)
    monkeypatch.setattr("fastvideo.worker.gpu_worker.Worker.init_device", lambda self: None)
    executor = UniprocExecutor(FastVideoArgs(model_path="test", num_gpus=1))
    try:
        assert executor.driver_worker is not None
        assert executor.driver_worker.worker is not None
    finally:
        executor.shutdown()


class _FakeWorker:

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def execute_method(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        if method == "execute_forward":
            return ForwardBatch(data_type="video", output=None, extra={})
        if method == "set_lora_adapter":
            return {"status": "lora_adapter_set"}
        if method == "unmerge_lora_weights":
            return {"status": "lora_adapter_unmerged"}
        if method == "merge_lora_weights":
            return {"status": "lora_adapter_merged"}
        if method == "shutdown":
            return {"status": "shutdown_complete"}
        return {"status": "ok"}

    def shutdown(self):
        return self.execute_method("shutdown")


def _executor_with_fake_worker() -> tuple[UniprocExecutor, _FakeWorker]:
    worker = _FakeWorker()
    executor = UniprocExecutor.__new__(UniprocExecutor)
    executor.fastvideo_args = FastVideoArgs(model_path="test")
    executor.driver_worker = worker
    executor.shutting_down = False
    executor._log_queue = None
    executor._log_queue_handler = None
    return executor, worker


def test_collective_rpc_calls_local_worker() -> None:
    executor, worker = _executor_with_fake_worker()
    result = executor.collective_rpc("ping", args=(1, ), kwargs={"k": 2})
    assert result == [{"status": "ok"}]
    assert worker.calls == [("ping", (1, ), {"k": 2})]


def test_lora_and_shutdown_go_through_local_rpc() -> None:
    executor, worker = _executor_with_fake_worker()
    executor.set_lora_adapter("default", "/weights", strength=0.5)
    executor.merge_lora_weights()
    executor.unmerge_lora_weights()
    executor.shutdown()
    methods = [call[0] for call in worker.calls]
    assert methods == ["set_lora_adapter", "merge_lora_weights", "unmerge_lora_weights", "shutdown"]


def test_execute_forward_returns_worker_batch(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    executor, worker = _executor_with_fake_worker()
    batch = ForwardBatch(data_type="video")
    args = FastVideoArgs(model_path="test")
    result = executor.execute_forward(batch, args)
    assert result.extra == {}
    assert worker.calls[0][0] == "execute_forward"


def test_uniproc_log_queue_stays_in_process() -> None:
    executor, _worker = _executor_with_fake_worker()
    queue = SimpleNamespace()
    executor.set_log_queue(queue)
    assert executor._log_queue is queue
    assert executor._log_queue_handler is not None
    executor.clear_log_queue()
    assert executor._log_queue is None
    assert executor._log_queue_handler is None
    assert "log_queue" in signature(Executor.set_log_queue).parameters
