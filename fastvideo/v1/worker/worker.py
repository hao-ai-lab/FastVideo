from abc import ABC

import torch

from typing import Optional

from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ForwardBatch
from fastvideo.v1.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

logger = init_logger(__name__)


class WorkerBase(ABC):

    def __init__(self, inference_args: InferenceArgs):
        self.inference_args = inference_args

    def execute_forward(self, forward_batch: ForwardBatch) -> torch.Tensor:
        pass

    def start_worker_execution_loop(self) -> None:
        pass


class Worker(WorkerBase):

    def __init__(self, inference_args: InferenceArgs):
        super().__init__(inference_args)

    def execute_forward(self, forward_batch: ForwardBatch) -> torch.Tensor:
        pass


def init_worker_distributed_environment(
    inference_args: InferenceArgs,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize distributed environment and model parallelism."""

    world_size = inference_args.num_gpus

    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 local_rank=local_rank)
    device_str = f"cuda:{local_rank}"
    inference_args.device_str = device_str
    inference_args.device = torch.device(device_str)
    assert inference_args.sp_size is not None
    assert inference_args.tp_size is not None
    initialize_model_parallel(
        sequence_model_parallel_size=inference_args.sp_size,
        tensor_model_parallel_size=inference_args.tp_size,
    )
