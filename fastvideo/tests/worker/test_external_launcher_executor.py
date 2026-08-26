# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the external-launcher inference executor."""

import os
from pathlib import Path
import subprocess
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
from fastvideo.worker.gpu_worker import Worker
from fastvideo.worker.multiproc_executor import MultiprocExecutor

TORCHRUN_ENV = {
    "RANK": "5",
    "LOCAL_RANK": "1",
    "LOCAL_WORLD_SIZE": "4",
    "WORLD_SIZE": "8",
    "MASTER_ADDR": "10.0.0.1",
    "MASTER_PORT": "29500",
}


def _clear_launcher_env(monkeypatch):
    for name in (
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_LOCALID",
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_STEP_TASKS_PER_NODE",
        "SLURM_TASKS_PER_NODE",
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


def test_resolve_native_slurm_identity_mapping():
    context = resolve_external_launcher_env({
        "SLURM_PROCID": "5",
        "SLURM_LOCALID": "1",
        "SLURM_NTASKS": "8",
        "SLURM_NTASKS_PER_NODE": "4(x2)",
        "MASTER_ADDR": "10.0.0.1",
        "MASTER_PORT": "29500",
    })

    assert (context.rank, context.local_rank, context.world_size) == (5, 1, 8)
    assert context.local_world_size == 4


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


@pytest.mark.parametrize("master_port", ["0", "65536", "not-a-port"])
def test_resolve_rejects_invalid_master_port(master_port):
    environ = dict(TORCHRUN_ENV, MASTER_PORT=master_port)

    with pytest.raises(ValueError, match="MASTER_PORT"):
        resolve_external_launcher_env(environ)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RANK", "not-a-rank"),
        ("WORLD_SIZE", "many"),
        ("LOCAL_RANK", "local"),
        ("LOCAL_WORLD_SIZE", "several"),
    ],
)
def test_resolve_rejects_non_integer_identity_with_field_name(name, value):
    environ = dict(TORCHRUN_ENV, **{name: value})

    with pytest.raises(ValueError, match=name):
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

    assert Executor.get_class(args, allow_external_launcher=True) is ExternalLauncherExecutor


def test_get_class_explicit_backend_selects_external_launcher(monkeypatch):
    _clear_launcher_env(monkeypatch)
    args = FastVideoArgs(model_path="test", distributed_executor_backend="external_launcher")

    assert Executor.get_class(args, allow_external_launcher=True) is ExternalLauncherExecutor


def test_get_class_rejects_external_launcher_without_spmd_opt_in(monkeypatch):
    _clear_launcher_env(monkeypatch)
    args = FastVideoArgs(model_path="test", distributed_executor_backend="external_launcher")

    with pytest.raises(ValueError, match="synchronized offline generation"):
        Executor.get_class(args)


def test_streaming_generator_rejects_external_launcher(monkeypatch):
    _clear_launcher_env(monkeypatch)
    from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
    args = FastVideoArgs(model_path="test", distributed_executor_backend="external_launcher")

    with pytest.raises(ValueError, match="StreamingVideoGenerator"):
        StreamingVideoGenerator.from_fastvideo_args(args)


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
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    args = FastVideoArgs(model_path="test", num_gpus=8, sp_size=8)

    executor = ExternalLauncherExecutor(args)

    assert os.environ["LOCAL_RANK"] == "3"
    assert os.environ["RANK"] == "5"
    assert os.environ["WORLD_SIZE"] == "8"
    assert not args.is_output_rank
    assert wrappers[0].local_rank_at_init_device == "3"
    assert wrappers[0].init_kwargs == {
        "fastvideo_args": args,
        "local_rank": 3,
        "rank": 5,
        "distributed_init_method": "env://",
    }
    executor.shutdown()
    assert wrappers[0].shutdown_called


def test_init_rejects_local_rank_outside_visible_cuda_devices(monkeypatch):
    _clear_launcher_env(monkeypatch)
    environ = dict(TORCHRUN_ENV, LOCAL_RANK="3")
    _set_env(monkeypatch, environ)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    args = FastVideoArgs(model_path="test", num_gpus=8, sp_size=8)

    with pytest.raises(ValueError, match="process-visible CUDA"):
        ExternalLauncherExecutor(args)


def test_init_failure_cleans_up_partially_initialized_worker(monkeypatch):
    _clear_launcher_env(monkeypatch)
    _set_env(monkeypatch, TORCHRUN_ENV)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    shutdown = Mock()

    class FailingWorkerWrapper:

        def __init__(self, fastvideo_args, rpc_rank):
            self.fastvideo_args = fastvideo_args
            self.rpc_rank = rpc_rank
            self.worker = SimpleNamespace()

        def init_worker(self, all_kwargs):
            self.kwargs = all_kwargs[self.rpc_rank]

        def init_device(self):
            raise RuntimeError("injected device initialization failure")

        def shutdown(self):
            shutdown()

    monkeypatch.setattr(external_launcher, "WorkerWrapperBase", FailingWorkerWrapper)
    args = FastVideoArgs(model_path="test", num_gpus=8, sp_size=8)

    with pytest.raises(RuntimeError, match="injected device initialization failure"):
        ExternalLauncherExecutor(args)
    shutdown.assert_called_once_with()


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
    assert executor.uses_spmd_execution
    assert MultiprocExecutor.__new__(MultiprocExecutor).is_output_rank


def test_collective_rpc_rejects_unsupported_timeout():
    executor = ExternalLauncherExecutor.__new__(ExternalLauncherExecutor)
    executor.rank = 0
    executor.world_size = 1
    executor.shutting_down = True
    executor.worker = SimpleNamespace(execute_method=Mock())

    with pytest.raises(NotImplementedError, match="does not support per-call timeouts"):
        executor.collective_rpc("probe", timeout=0.01)
    executor.worker.execute_method.assert_not_called()


def test_collective_rpc_single_rank_returns_rank_ordered_result(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    executor = ExternalLauncherExecutor.__new__(ExternalLauncherExecutor)
    executor.rank = 0
    executor.world_size = 1
    executor.shutting_down = True
    executor.worker = SimpleNamespace(
        device=torch.device("cpu"),
        execute_method=lambda method, *args, **kwargs: {
            "method": method,
            "args": args,
            "kwargs": kwargs,
        },
    )

    assert executor.collective_rpc("probe", args=(1, ), kwargs={"two": 2}) == [{
        "method": "probe",
        "args": (1, ),
        "kwargs": {
            "two": 2
        },
    }]


def test_worker_non_output_rank_clears_latents_audio_and_trajectories():
    args = SimpleNamespace(is_output_rank=False, output_type="latent")
    batch = external_launcher.ForwardBatch(
        data_type="video",
        output=torch.ones(1),
        latents=torch.ones(1),
        audio_latents=torch.ones(1),
        save_video=True,
        return_frames=True,
        return_trajectory_latents=True,
        return_trajectory_decoded=True,
        extra={
            "audio": torch.ones(1),
            "audio_sample_rate": 24000,
            "decoded_audio": torch.ones(1),
            "ltx2_audio_latents": torch.ones(1),
        },
        trajectory_latents=torch.ones(1),
        trajectory_timesteps=[torch.ones(1)],
        trajectory_decoded=[torch.ones(1)],
    )
    worker = Worker.__new__(Worker)
    worker.fastvideo_args = args
    worker.pipeline = SimpleNamespace(forward=lambda forward_batch, fastvideo_args: forward_batch)

    result = worker.execute_forward(batch, args)

    assert not batch.save_video
    assert not batch.return_frames
    assert not batch.return_trajectory_latents
    assert not batch.return_trajectory_decoded
    assert result.output is not None and result.output.numel() == 0
    assert result.latents is None
    assert result.audio_latents is None
    assert "audio" not in result.extra
    assert "audio_sample_rate" not in result.extra
    assert "decoded_audio" not in result.extra
    assert "ltx2_audio_latents" not in result.extra
    assert result.trajectory_latents is None
    assert result.trajectory_timesteps is None
    assert result.trajectory_decoded is None


@pytest.mark.parametrize("mode", ["success", "rank-failure"])
def test_external_launcher_real_two_process_lifecycle(mode):
    helper = Path(__file__).with_name("external_launcher_lifecycle_smoke.py")
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["FASTVIDEO_TARGET_DEVICE"] = "cpu"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(helper),
            "--mode",
            mode,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    marker = "lifecycle_ok" if mode == "success" else "coordinated_failure"
    assert completed.stdout.count(marker) == 2


def test_external_launcher_real_two_process_peer_exit_is_bounded():
    helper = Path(__file__).with_name("external_launcher_lifecycle_smoke.py")
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["FASTVIDEO_TARGET_DEVICE"] = "cpu"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(helper),
            "--mode",
            "rank-exit",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "injected pre-request rank exit" in completed.stderr


def test_external_launcher_real_two_process_missing_collective_honors_dist_timeout():
    helper = Path(__file__).with_name("external_launcher_lifecycle_smoke.py")
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["FASTVIDEO_TARGET_DEVICE"] = "cpu"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(helper),
            "--mode",
            "collective-timeout",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "collective_timeout rank=0" in completed.stdout, completed.stdout + completed.stderr
