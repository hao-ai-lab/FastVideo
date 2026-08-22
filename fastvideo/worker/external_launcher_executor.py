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
from typing import Any

import torch

import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.worker.executor import Executor
from fastvideo.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

EXTERNAL_LAUNCHER_BACKEND = "external_launcher"
_REQUIRED_LAUNCHER_VARS = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@dataclass(frozen=True)
class ExternalLauncherEnv:
    """Distributed identity assigned to this process by its launcher."""

    rank: int
    local_rank: int
    world_size: int


def resolve_external_launcher_env(environ: Mapping[str, str]) -> ExternalLauncherEnv:
    """Parse and validate a torchrun/srun distributed environment."""

    missing = [name for name in _REQUIRED_LAUNCHER_VARS if not environ.get(name)]
    if missing:
        raise RuntimeError("External-launcher inference requires the launcher to provide "
                           f"{missing}. Launch with torchrun/srun and an env:// rendezvous.")

    rank = int(environ["RANK"])
    world_size = int(environ["WORLD_SIZE"])
    local_rank_str = environ.get("LOCAL_RANK") or environ.get("SLURM_LOCALID")
    if local_rank_str is None:
        if world_size == 1:
            local_rank_str = "0"
        else:
            raise RuntimeError("External-launcher inference requires LOCAL_RANK (torchrun) "
                               "or SLURM_LOCALID (srun) so each process can select its GPU.")
    local_rank = int(local_rank_str)

    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE); got RANK={rank}, WORLD_SIZE={world_size}.")
    if not 0 <= local_rank < world_size:
        raise ValueError(
            f"LOCAL_RANK must be in [0, WORLD_SIZE); got LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}.")
    return ExternalLauncherEnv(rank=rank, local_rank=local_rank, world_size=world_size)


class ExternalLauncherExecutor(Executor):
    """Run this externally launched process's single worker inline."""

    def _init_executor(self) -> None:
        env_ctx = resolve_external_launcher_env(os.environ)
        if self.fastvideo_args.num_gpus != env_ctx.world_size:
            raise ValueError(
                f"num_gpus={self.fastvideo_args.num_gpus} does not match the external launcher's "
                f"WORLD_SIZE={env_ctx.world_size}. Pass num_gpus equal to the total number of launched processes.")

        # get_local_torch_device() reads LOCAL_RANK through fastvideo.envs.
        # torchrun already provides it; normalize SLURM_LOCALID for native
        # srun launchers before Worker.init_device() performs any CUDA work.
        os.environ["LOCAL_RANK"] = str(env_ctx.local_rank)

        # Downstream Worker initialization must preserve the launcher's
        # global rank/world rather than deriving a local multiprocess layout.
        self.fastvideo_args.distributed_executor_backend = EXTERNAL_LAUNCHER_BACKEND
        self.rank = env_ctx.rank
        self.world_size = env_ctx.world_size
        self.shutting_down = False

        logger.info(
            "External-launcher executor: rank=%d local_rank=%d world_size=%d "
            "(env:// rendezvous at %s:%s)",
            env_ctx.rank,
            env_ctx.local_rank,
            env_ctx.world_size,
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
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
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper
        wrapper.init_device()

        atexit.register(self.shutdown)

    @property
    def is_output_rank(self) -> bool:
        return self.rank == 0

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

        return ForwardBatch(
            data_type=forward_batch.data_type,
            output=output_batch.output,
            logging_info=logging_info,
            extra=extra,
            trajectory_latents=output_batch.trajectory_latents,
            trajectory_timesteps=output_batch.trajectory_timesteps,
        )

    def collective_rpc(
            self,
            method: str | Callable,
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict | None = None,
    ) -> list[Any]:
        """Execute one local worker method in every SPMD process."""

        del timeout
        self._bind_worker_device()
        return [self.worker.execute_method(method, *args, **(kwargs or {}))]

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
        if responses[0]["status"] != "lora_adapter_set":
            raise RuntimeError(f"Rank {self.rank} failed to set LoRA adapter to {lora_path}")

    def unmerge_lora_weights(self) -> None:
        responses = self.collective_rpc("unmerge_lora_weights")
        if responses[0]["status"] != "lora_adapter_unmerged":
            raise RuntimeError(f"Rank {self.rank} failed to unmerge LoRA weights")

    def merge_lora_weights(self) -> None:
        responses = self.collective_rpc("merge_lora_weights")
        if responses[0]["status"] != "lora_adapter_merged":
            raise RuntimeError(f"Rank {self.rank} failed to merge LoRA weights")

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
