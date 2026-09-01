# SPDX-License-Identifier: Apache-2.0
from inspect import signature

from fastvideo.worker.executor import Executor
from fastvideo.worker.ray_distributed_executor import RayDistributedExecutor


def test_ray_executor_implements_executor_abc() -> None:
    remaining = getattr(RayDistributedExecutor, "__abstractmethods__", frozenset())
    assert remaining == frozenset(), remaining


def test_ray_log_queue_stays_on_the_driver() -> None:
    """multiprocessing.Queue cannot be pickled onto a remote Ray worker."""
    executor = RayDistributedExecutor.__new__(RayDistributedExecutor)
    executor.set_log_queue(object())
    assert executor._log_queue is not None
    executor.clear_log_queue()
    assert executor._log_queue is None
    assert "log_queue" in signature(Executor.set_log_queue).parameters
