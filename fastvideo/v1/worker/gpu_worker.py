from typing import Optional
from fastvideo.v1.worker.worker_base import WorkerBase
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
import torch
import os
import gc
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines import build_pipeline
from fastvideo.v1.distributed import (init_distributed_environment,
                                      initialize_model_parallel)
from fastvideo.v1.platforms import current_platform

logger = init_logger(__name__)


class Worker(WorkerBase):

    def __init__(self,
                 fastvideo_args: FastVideoArgs,
                 local_rank: int,
                 rank: int,
                 distributed_init_method: str,
                 is_driver_worker: bool = False):
        super().__init__(fastvideo_args, local_rank, rank,
                         distributed_init_method, is_driver_worker)

        # Torch profiler. Enabled and configured through env vars:
        # FASTVIDEO_TORCH_PROFILER_DIR=/path/to/save/trace

        # TODO: enable profiling
        # if envs.FASTVIDEO_TORCH_PROFILER_DIR:
        if False:
            # torch_profiler_trace_dir = envs.FASTVIDEO_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def init_device(self):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.fastvideo_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        # set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.pipeline: Optional[ComposedPipelineBase] = None

    def load_pipeline(self) -> None:
        self.pipeline = build_pipeline(self.fastvideo_config)
        logger.info("Pipeline Ready")

    def get_pipeline(self) -> ComposedPipelineBase:
        return self.pipeline

    @torch.inference_mode()
    def execute_model(
        self,
        forward_batch: ForwardBatch,
    ) -> Optional[ForwardBatch]:
        output = self.pipeline.forward(forward_batch)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    # def add_lora(self, lora_request: LoRARequest) -> bool:
    #     return self.model_runner.add_lora(lora_request)

    # def remove_lora(self, lora_id: int) -> bool:
    #     return self.model_runner.remove_lora(lora_id)

    # def list_loras(self) -> set[int]:
    #     return self.model_runner.list_loras()

    # def pin_lora(self, lora_id: int) -> bool:
    #     return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return


def init_worker_distributed_environment(
    fastvideo_args: FastVideoArgs,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(fastvideo_args.world_size, rank,
                                 distributed_init_method, local_rank)

    # ensure_model_parallel_initialized(fastvideo_args.tp_size,
    #                                   fastvideo_args.sp_size)
    initialize_model_parallel(
        sequence_model_parallel_size=fastvideo_args.sp_size,
        tensor_model_parallel_size=fastvideo_args.tp_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the "
                "`dtype` flag in CLI, for example: --dtype=half.")
