# SPDX-License-Identifier: Apache-2.0
"""In-process executor for externally launched SPMD inference.

Unlike :class:`MultiprocExecutor`, which starts local subprocesses, this
executor turns each process started by torchrun or srun into one inline
worker. The launcher owns ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``,
``MASTER_ADDR``, and ``MASTER_PORT``; FastVideo initializes through the
``env://`` rendezvous.

Every rank must call generation collectively with the same requests in the
same order. World rank 0 owns saved media and returned frames; other ranks
still execute the full pipeline so sequence-parallel collectives stay
uniform.
"""

from __future__ import annotations

import atexit
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from queue import Queue
import re
from typing import Any

import torch

import fastvideo.envs as envs
from fastvideo.distributed import get_world_group
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.worker.executor import EXTERNAL_LAUNCHER_BACKEND, Executor
from fastvideo.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

_NON_OUTPUT_EXTRA_KEYS = frozenset({
    "audio",
    "audio_sample_rate",
    "decoded_audio",
    "ltx2_audio_latents",
})


@dataclass(frozen=True)
class ExternalLauncherEnv:
    """Distributed identity assigned to this process by its launcher."""

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int | None
    master_addr: str
    master_port: int


def _first_env_value(environ: Mapping[str, str], *names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = environ.get(name)
        if value:
            return value, name
    return None, None


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from None


def _slurm_local_world_size(environ: Mapping[str, str]) -> int | None:
    """Resolve a uniform Slurm tasks-per-node value when one is available."""
    value, source = _first_env_value(
        environ,
        "SLURM_NTASKS_PER_NODE",
        "SLURM_STEP_TASKS_PER_NODE",
        "SLURM_TASKS_PER_NODE",
    )
    if value is None or source is None:
        return None
    match = re.fullmatch(r"\s*(\d+)(?:\(x\d+\))?\s*", value)
    if match is None:
        # Heterogeneous encodings such as ``4(x2),2`` do not identify the
        # current node's task count. Device-count validation still applies.
        return None
    return _parse_int(source, match.group(1))


def resolve_external_launcher_env(environ: Mapping[str, str]) -> ExternalLauncherEnv:
    """Parse and validate a torchrun/srun distributed environment."""

    rank_value, rank_source = _first_env_value(environ, "RANK", "SLURM_PROCID")
    world_size_value, world_size_source = _first_env_value(environ, "WORLD_SIZE", "SLURM_NTASKS")
    master_addr, _ = _first_env_value(environ, "MASTER_ADDR")
    master_port_value, _ = _first_env_value(environ, "MASTER_PORT")
    missing = []
    if rank_value is None:
        missing.append("RANK (or SLURM_PROCID)")
    if world_size_value is None:
        missing.append("WORLD_SIZE (or SLURM_NTASKS)")
    if master_addr is None:
        missing.append("MASTER_ADDR")
    if master_port_value is None:
        missing.append("MASTER_PORT")
    if missing:
        raise RuntimeError("External-launcher inference requires the launcher to provide "
                           f"{missing}. Launch with torchrun/srun and an env:// rendezvous.")

    assert rank_value is not None and rank_source is not None
    assert world_size_value is not None and world_size_source is not None
    assert master_addr is not None and master_port_value is not None
    rank = _parse_int(rank_source, rank_value)
    world_size = _parse_int(world_size_source, world_size_value)
    master_port = _parse_int("MASTER_PORT", master_port_value)

    local_rank_value, local_rank_source = _first_env_value(environ, "LOCAL_RANK", "SLURM_LOCALID")
    if local_rank_value is None:
        if world_size == 1:
            local_rank = 0
        else:
            raise RuntimeError("External-launcher inference requires LOCAL_RANK (torchrun) "
                               "or SLURM_LOCALID (srun) so each process can select its GPU.")
    else:
        assert local_rank_source is not None
        local_rank = _parse_int(local_rank_source, local_rank_value)

    local_world_size_value, local_world_size_source = _first_env_value(environ, "LOCAL_WORLD_SIZE")
    local_world_size: int | None
    if local_world_size_value is not None:
        assert local_world_size_source is not None
        local_world_size = _parse_int(local_world_size_source, local_world_size_value)
    else:
        local_world_size = _slurm_local_world_size(environ)

    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE); got RANK={rank}, WORLD_SIZE={world_size}.")
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}.")
    if local_world_size is not None:
        if local_world_size < 1:
            raise ValueError(f"LOCAL_WORLD_SIZE must be >= 1, got {local_world_size}.")
        if local_rank >= local_world_size:
            raise ValueError("LOCAL_RANK must be in [0, LOCAL_WORLD_SIZE); "
                             f"got LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={local_world_size}.")
    elif local_rank >= world_size:
        raise ValueError(
            f"LOCAL_RANK must be in [0, WORLD_SIZE); got LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}.")
    if not 1 <= master_port <= 65535:
        raise ValueError(f"MASTER_PORT must be in [1, 65535], got {master_port}.")
    return ExternalLauncherEnv(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        master_addr=master_addr,
        master_port=master_port,
    )


class ExternalLauncherExecutor(Executor):
    """Run this externally launched process's single worker inline."""

    def _init_executor(self) -> None:
        env_ctx = resolve_external_launcher_env(os.environ)
        if self.fastvideo_args.num_gpus != env_ctx.world_size:
            raise ValueError(
                f"num_gpus={self.fastvideo_args.num_gpus} does not match the external launcher's "
                f"WORLD_SIZE={env_ctx.world_size}. Pass num_gpus equal to the total number of launched processes.")

        if torch.cuda.is_available():
            visible_device_count = torch.cuda.device_count()
            if env_ctx.local_rank >= visible_device_count:
                raise ValueError(
                    f"LOCAL_RANK={env_ctx.local_rank} cannot select from the {visible_device_count} "
                    "process-visible CUDA device(s). Configure launcher GPU visibility so each process can address "
                    "its local rank before model loading.")

        # get_local_torch_device() reads LOCAL_RANK through fastvideo.envs.
        # Normalize both torchrun and native Slurm names before
        # Worker.init_device() performs any accelerator or process-group work.
        os.environ["RANK"] = str(env_ctx.rank)
        os.environ["WORLD_SIZE"] = str(env_ctx.world_size)
        os.environ["LOCAL_RANK"] = str(env_ctx.local_rank)
        if env_ctx.local_world_size is not None:
            os.environ["LOCAL_WORLD_SIZE"] = str(env_ctx.local_world_size)

        # Downstream Worker initialization must preserve the launcher's
        # global rank/world rather than deriving a local multiprocess layout.
        self.fastvideo_args.distributed_executor_backend = EXTERNAL_LAUNCHER_BACKEND
        self.fastvideo_args.is_output_rank = env_ctx.rank == 0
        self.rank = env_ctx.rank
        self.world_size = env_ctx.world_size
        self.shutting_down = False

        logger.info(
            "External-launcher executor: rank=%d local_rank=%d world_size=%d "
            "(env:// rendezvous at %s:%s)",
            env_ctx.rank,
            env_ctx.local_rank,
            env_ctx.world_size,
            env_ctx.master_addr,
            env_ctx.master_port,
            local_main_process_only=False,
        )

        wrapper = WorkerWrapperBase(fastvideo_args=self.fastvideo_args, rpc_rank=env_ctx.rank)
        all_kwargs: list[dict[str, Any]] = [{} for _ in range(env_ctx.world_size)]
        all_kwargs[env_ctx.rank] = {
            "fastvideo_args": self.fastvideo_args,
            "local_rank": env_ctx.local_rank,
            "rank": env_ctx.rank,
            "distributed_init_method": "env://",
        }
        self.worker = wrapper
        try:
            wrapper.init_worker(all_kwargs)
            wrapper.init_device()
        except BaseException:
            self.shutting_down = True
            if getattr(wrapper, "worker", None) is not None:
                try:
                    wrapper.shutdown()
                except Exception:
                    logger.exception("Failed to clean up rank %d after launcher initialization error", self.rank)
            raise

        atexit.register(self.shutdown)

    @property
    def is_output_rank(self) -> bool:
        return self.rank == 0

    @property
    def uses_spmd_execution(self) -> bool:
        return True

    def broadcast_from_output_rank(self, value):
        """Broadcast a small control-plane value from global rank zero."""
        return get_world_group().broadcast_object(value, src=0)

    def _bind_worker_device(self) -> None:
        """Bind the request thread to this worker's CUDA device.

        VideoGenerator executes forwards in a fresh thread. CUDA device
        selection is thread-local, so the inline executor must repeat the
        binding performed by Worker.init_device() before request-thread CUDA
        work begins.
        """

        device = getattr(self.worker, "device", None)
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(device)

    def execute_forward(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        self._bind_worker_device()
        output_batch = self.worker.execute_forward(forward_batch, fastvideo_args)

        logging_info = output_batch.logging_info if envs.FASTVIDEO_STAGE_LOGGING else None
        extra = output_batch.extra or {}
        if torch.cuda.is_available():
            extra["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if not self.is_output_rank:
            for key in _NON_OUTPUT_EXTRA_KEYS:
                extra.pop(key, None)
            output_batch.output = torch.empty(0, device="cpu")
            output_batch.latents = None
            output_batch.audio_latents = None
            output_batch.trajectory_latents = None
            output_batch.trajectory_timesteps = None
            output_batch.trajectory_decoded = None

        return ForwardBatch(
            data_type=forward_batch.data_type,
            output=output_batch.output,
            logging_info=logging_info,
            extra=extra,
            trajectory_latents=output_batch.trajectory_latents,
            trajectory_timesteps=output_batch.trajectory_timesteps,
            trajectory_decoded=output_batch.trajectory_decoded,
        )

    def collective_rpc(
            self,
            method: str | Callable,
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict | None = None,
    ) -> list[Any]:
        """Execute a control call locally and return rank-ordered world results.

        Every launcher process must enter this method with the same call. A
        local exception is gathered before any rank raises, so ordinary
        control-plane failures are reported consistently instead of leaving a
        successful peer behind. Per-call timeouts are unsupported; the
        process-group timeout and launcher failure policy bound hard failures.
        """

        if timeout is not None:
            raise NotImplementedError(
                "ExternalLauncherExecutor.collective_rpc does not support per-call timeouts; "
                "configure the distributed process-group timeout and launcher kill-on-failure policy instead.")
        self._bind_worker_device()
        try:
            response = self.worker.execute_method(method, *args, **(kwargs or {}))
            local_status: dict[str, Any] = {
                "ok": True,
                "rank": self.rank,
                "response": response,
            }
        except Exception as error:
            local_status = {
                "ok": False,
                "rank": self.rank,
                "error_type": type(error).__name__,
                "error": str(error),
            }

        statuses: list[dict[str, Any] | None]
        if self.world_size == 1:
            statuses = [local_status]
        else:
            statuses = [None] * self.world_size
            torch.distributed.all_gather_object(statuses, local_status, group=get_world_group().cpu_group)

        failures = [status for status in statuses if status is not None and not status["ok"]]
        if failures:
            details = "; ".join(f"rank {failure['rank']} {failure['error_type']}: {failure['error']}"
                                for failure in failures)
            method_name = method if isinstance(method, str) else getattr(method, "__name__", repr(method))
            raise RuntimeError(f"Collective RPC {method_name!r} failed: {details}")
        return [status["response"] for status in statuses if status is not None]

    def set_lora_adapter(
        self,
        lora_nickname: str,
        lora_path: str | None = None,
        strength: float = 1.0,
        accumulate: bool = False,
    ) -> None:
        responses = self.collective_rpc(
            "set_lora_adapter",
            kwargs={
                "lora_nickname": lora_nickname,
                "lora_path": lora_path,
                "strength": strength,
                "accumulate": accumulate,
            },
        )
        for rank, response in enumerate(responses):
            if response["status"] != "lora_adapter_set":
                raise RuntimeError(f"Rank {rank} failed to set LoRA adapter to {lora_path}")

    def unmerge_lora_weights(self) -> None:
        responses = self.collective_rpc("unmerge_lora_weights")
        for rank, response in enumerate(responses):
            if response["status"] != "lora_adapter_unmerged":
                raise RuntimeError(f"Rank {rank} failed to unmerge LoRA weights")

    def merge_lora_weights(self) -> None:
        responses = self.collective_rpc("merge_lora_weights")
        for rank, response in enumerate(responses):
            if response["status"] != "lora_adapter_merged":
                raise RuntimeError(f"Rank {rank} failed to merge LoRA weights")

    def set_log_queue(self, log_queue: Queue | None) -> None:
        # The worker is in this process, so its logs already reach the local
        # handlers and do not need forwarding through a multiprocessing queue.
        del log_queue

    def clear_log_queue(self) -> None:
        pass

    def shutdown(self) -> None:
        if getattr(self, "shutting_down", False):
            return
        self.shutting_down = True
        worker = getattr(self, "worker", None)
        if worker is not None:
            worker.shutdown()

    def __del__(self) -> None:
        self.shutdown()
