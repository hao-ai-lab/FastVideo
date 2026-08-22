# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the external-launcher inference executor."""

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import fastvideo.worker.external_launcher_executor as external_launcher
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.worker.executor import Executor
from fastvideo.worker.external_launcher_executor import (
    ExternalLauncherExecutor,
    resolve_external_launcher_env,
)
from fastvideo.worker.multiproc_executor import MultiprocExecutor

TORCHRUN_ENV = {
    "RANK": "5",
    "LOCAL_RANK": "1",
    "WORLD_SIZE": "8",
    "MASTER_ADDR": "10.0.0.1",
    "MASTER_PORT": "29500",
}


def _clear_launcher_env(monkeypatch):
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_LOCALID",
        "FASTVIDEO_EXTERNAL_LAUNCHER",
    ):
        monkeypatch.delenv(name, raising=False)


def _set_env(monkeypatch, environ):
    for name, value in environ.items():
        monkeypatch.setenv(name, value)


def test_resolve_torchrun_env(monkeypatch):
    _clear_launcher_env(monkeypatch)
    _set_env(monkeypatch, TORCHRUN_ENV)

    context = resolve_external_launcher_env(os.environ)

    assert (context.rank, context.local_rank, context.world_size) == (5, 1, 8)


def test_resolve_falls_back_to_slurm_localid(monkeypatch):
    _clear_launcher_env(monkeypatch)
    environ = dict(TORCHRUN_ENV)
    del environ["LOCAL_RANK"]
    environ["SLURM_LOCALID"] = "3"
    _set_env(monkeypatch, environ)

    context = resolve_external_launcher_env(os.environ)

    assert context.local_rank == 3


def test_resolve_single_process_defaults_local_rank():
    environ = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }

    context = resolve_external_launcher_env(environ)

    assert (context.rank, context.local_rank, context.world_size) == (0, 0, 1)


def test_resolve_missing_vars_lists_them():
    with pytest.raises(RuntimeError) as excinfo:
        resolve_external_launcher_env({"RANK": "0", "WORLD_SIZE": "4"})

    message = str(excinfo.value)
    assert "MASTER_ADDR" in message
    assert "MASTER_PORT" in message
    assert "torchrun/srun" in message


def test_resolve_missing_local_rank_for_multiple_processes_errors():
    environ = {
        "RANK": "2",
        "WORLD_SIZE": "4",
        "MASTER_ADDR": "10.0.0.1",
        "MASTER_PORT": "29500",
    }

    with pytest.raises(RuntimeError, match="LOCAL_RANK"):
        resolve_external_launcher_env(environ)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RANK", "8"),
        ("RANK", "-1"),
        ("WORLD_SIZE", "0"),
        ("LOCAL_RANK", "8"),
        ("LOCAL_RANK", "-1"),
    ],
)
def test_resolve_rejects_out_of_bounds_identity(name, value):
    environ = dict(TORCHRUN_ENV, **{name: value})

    with pytest.raises(ValueError, match=name):
        resolve_external_launcher_env(environ)


def test_get_class_default_stays_multiproc(monkeypatch):
    _clear_launcher_env(monkeypatch)
    args = FastVideoArgs(model_path="test")

    assert Executor.get_class(args) is MultiprocExecutor


def test_get_class_env_flag_selects_external_launcher(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("FASTVIDEO_EXTERNAL_LAUNCHER", "1")
    args = FastVideoArgs(model_path="test")

    assert Executor.get_class(args) is ExternalLauncherExecutor


def test_get_class_explicit_backend_selects_external_launcher(monkeypatch):
    _clear_launcher_env(monkeypatch)
    args = FastVideoArgs(model_path="test", distributed_executor_backend="external_launcher")

    assert Executor.get_class(args) is ExternalLauncherExecutor


def test_env_flag_does_not_override_explicit_ray_backend(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("FASTVIDEO_EXTERNAL_LAUNCHER", "1")

    class StubRayExecutor:
        pass

    ray_module = ModuleType("fastvideo.worker.ray_distributed_executor")
    ray_module.RayDistributedExecutor = StubRayExecutor
    monkeypatch.setitem(sys.modules, ray_module.__name__, ray_module)
    args = FastVideoArgs(model_path="test", distributed_executor_backend="ray")

    assert Executor.get_class(args) is StubRayExecutor


def test_init_rejects_num_gpus_world_size_mismatch(monkeypatch):
    _clear_launcher_env(monkeypatch)
    _set_env(monkeypatch, TORCHRUN_ENV)
    args = FastVideoArgs(model_path="test", num_gpus=4, sp_size=4)

    with pytest.raises(ValueError, match="WORLD_SIZE"):
        ExternalLauncherExecutor(args)


def test_init_requires_launcher_env(monkeypatch):
    _clear_launcher_env(monkeypatch)
    args = FastVideoArgs(model_path="test", num_gpus=8, sp_size=8)

    with pytest.raises(RuntimeError, match="torchrun/srun"):
        ExternalLauncherExecutor(args)


def test_init_normalizes_slurm_localid_before_worker_device_init(monkeypatch):
    _clear_launcher_env(monkeypatch)
    environ = dict(TORCHRUN_ENV)
    del environ["LOCAL_RANK"]
    environ["SLURM_LOCALID"] = "3"
    _set_env(monkeypatch, environ)

    wrappers = []

    class StubWorkerWrapper:

        def __init__(self, fastvideo_args, rpc_rank):
            self.fastvideo_args = fastvideo_args
            self.rpc_rank = rpc_rank
            self.init_kwargs = None
            self.local_rank_at_init_device = None
            self.shutdown_called = False
            wrappers.append(self)

        def init_worker(self, all_kwargs):
            self.init_kwargs = all_kwargs[self.rpc_rank]

        def init_device(self):
            self.local_rank_at_init_device = os.environ.get("LOCAL_RANK")

        def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setattr(external_launcher, "WorkerWrapperBase", StubWorkerWrapper)
    monkeypatch.setattr(external_launcher.atexit, "register", Mock())
    args = FastVideoArgs(model_path="test", num_gpus=8, sp_size=8)

    executor = ExternalLauncherExecutor(args)

    assert os.environ["LOCAL_RANK"] == "3"
    assert wrappers[0].local_rank_at_init_device == "3"
    assert wrappers[0].init_kwargs == {
        "fastvideo_args": args,
        "local_rank": 3,
        "rank": 5,
        "distributed_init_method": "env://",
    }
    executor.shutdown()
    assert wrappers[0].shutdown_called


def test_bind_worker_device_in_request_thread(monkeypatch):
    set_device = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    executor = ExternalLauncherExecutor.__new__(ExternalLauncherExecutor)
    executor.worker = SimpleNamespace(device=torch.device("cuda:2"), shutdown=Mock())

    executor._bind_worker_device()

    set_device.assert_called_once_with(torch.device("cuda:2"))
    executor.shutting_down = True


def test_is_output_rank_property():
    executor = ExternalLauncherExecutor.__new__(ExternalLauncherExecutor)
    executor.shutting_down = True
    executor.rank = 0
    assert executor.is_output_rank
    executor.rank = 3
    assert not executor.is_output_rank
    assert MultiprocExecutor.__new__(MultiprocExecutor).is_output_rank
