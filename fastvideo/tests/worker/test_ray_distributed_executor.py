# SPDX-License-Identifier: Apache-2.0
from inspect import signature

from fastvideo.worker.executor import Executor
from fastvideo.worker.ray_distributed_executor import (
    RayDistributedExecutor,
    should_use_gloo_loopback,
)


def test_ray_executor_implements_executor_abc() -> None:
    remaining = getattr(RayDistributedExecutor, "__abstractmethods__", frozenset())
    assert remaining == frozenset(), remaining


def test_gloo_loopback_follows_worker_ips_not_node_count() -> None:
    assert should_use_gloo_loopback(["192.168.23.2"]) is True
    assert should_use_gloo_loopback(["192.168.23.2", "192.168.23.2"]) is True
    assert should_use_gloo_loopback(["192.168.23.2", "192.168.23.1"]) is False


def test_ray_does_not_copy_per_node_nic_env_vars() -> None:
    nic = RayDistributedExecutor.WORKER_LOCAL_NIC_ENV_VARS
    assert "NCCL_SOCKET_IFNAME" in nic
    assert "NCCL_IB_HCA" in nic
    assert "GLOO_SOCKET_IFNAME" in nic
    copied = RayDistributedExecutor.ADDITIONAL_ENV_VARS
    assert not (nic & copied)


def test_ray_log_queue_stays_on_the_driver() -> None:
    """multiprocessing.Queue cannot be pickled onto a remote Ray worker."""
    executor = RayDistributedExecutor.__new__(RayDistributedExecutor)
    executor.set_log_queue(object())
    assert executor._log_queue is not None
    executor.clear_log_queue()
    assert executor._log_queue is None
    assert "log_queue" in signature(Executor.set_log_queue).parameters
