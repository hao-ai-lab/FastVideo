# SPDX-License-Identifier: Apache-2.0
from inspect import signature

import pytest

from fastvideo.worker.executor import Executor
from fastvideo.worker.ray_distributed_executor import (
    RAY_NOSET_CUDA_VISIBLE_DEVICES,
    RayDistributedExecutor,
    keep_raylet_cuda_devices,
    ray_worker_device_ordinal,
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


def test_ray_carries_h3_performance_switches() -> None:
    import fastvideo.envs as envs
    from fastvideo.worker.ray_env import get_env_vars_to_copy

    with (envs.FASTVIDEO_H3_VAE_TILE_BATCH.override(8),
          envs.FASTVIDEO_NVFP4_MM_BACKEND.override("cutlass"),
          envs.FASTVIDEO_VSA_TRITON.override(True)):
        copied = get_env_vars_to_copy()
        assert {"FASTVIDEO_H3_VAE_TILE_BATCH", "FASTVIDEO_NVFP4_MM_BACKEND", "FASTVIDEO_VSA_TRITON"} <= copied
        assert envs.FASTVIDEO_H3_VAE_TILE_BATCH.get() == 8
        assert envs.FASTVIDEO_NVFP4_MM_BACKEND.get() == "cutlass"
        assert envs.FASTVIDEO_VSA_TRITON.get() is True


def test_ray_actor_options_keep_the_raylet_device_list() -> None:
    options = {"runtime_env": {"env_vars": {"NCCL_DEBUG": "INFO"}, "pip": ["x"]}, "max_restarts": 0}
    kept = keep_raylet_cuda_devices(options)
    assert kept["runtime_env"]["env_vars"] == {"NCCL_DEBUG": "INFO", RAY_NOSET_CUDA_VISIBLE_DEVICES: "1"}
    assert kept["runtime_env"]["pip"] == ["x"]
    assert kept["max_restarts"] == 0
    # The caller's options are not modified.
    assert options["runtime_env"]["env_vars"] == {"NCCL_DEBUG": "INFO"}
    assert keep_raylet_cuda_devices({})["runtime_env"] == {"env_vars": {RAY_NOSET_CUDA_VISIBLE_DEVICES: "1"}}


def test_ray_worker_device_ordinal_addresses_the_inherited_device_list() -> None:
    # Raylet without CUDA_VISIBLE_DEVICES: Ray's GPU IDs are the ordinals.
    assert [ray_worker_device_ordinal([0, 1, 2, 3], i, None) for i in range(4)] == [0, 1, 2, 3]
    # Raylet started with CUDA_VISIBLE_DEVICES=2,3 (a second raylet on the same host).
    assert [ray_worker_device_ordinal([2, 3], i, "2,3") for i in range(2)] == [0, 1]
    # Four visible GPUs, of which this executor's workers on the node use 2 and 3.
    assert [ray_worker_device_ordinal([2, 3], i, "0,1,2,3") for i in range(2)] == [2, 3]
    with pytest.raises(RuntimeError, match="not in the worker's CUDA_VISIBLE_DEVICES"):
        ray_worker_device_ordinal([5], 0, "0,1")
