from abc import ABC

import torch

from typing import Optional, List
import multiprocessing as mp
import setproctitle
import psutil
import faulthandler
import signal
import traceback

from fastvideo.v1.fastvideo_args import FastVideoArgs, set_current_fastvideo_args
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ForwardBatch
from fastvideo.v1.utils import run_method, update_environment_variables
from fastvideo.v1.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
import zmq
from fastvideo.v1.utils import get_zmq_socket, kill_itself_when_parent_died, get_exception_traceback, TypeBasedDispatcher
from fastvideo.v1.worker.io_struct import RpcReqInput, RpcReqOutput, GenerateRequest


logger = init_logger(__name__)

# class WorkerWrapper5
#     def __init__(self, fastvideo_args: FastVideoArgs):
#         self.fastvideo_args = fastvideo_args
    
#     def update_environment_variables(self, envs_list: List[Dict[str, str]]) -> None:
#         envs = envs_list[self.rpc_rank]
#         key = 'CUDA_VISIBLE_DEVICES'
#         if key in envs and key in os.environ:
#             # overwriting CUDA_VISIBLE_DEVICES is desired behavior
#             # suppress the warning in `update_environment_variables`
#             del os.environ[key]
#         update_environment_variables(envs)
    
#     def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
#         kwargs = all_kwargs[self.rpc_rank]
#         self.fastvideo_args = kwargs.get("fastvideo_args", None)
#         assert self.fastvideo_args is not None, (
#             "fastvideo_args is required to initialize the worker")

#         with set_current_fastvideo_args(self.fastvideo_args):
#             self.worker = Worker(**kwargs)
#             assert self.worker is not None

#     def init_device(self):
#         self.worker.init_device()

#     def execute_method(self, method: Union[str, bytes], *args, **kwargs):
#         try:
#             # method resolution order:
#             # if a method is defined in this class, it will be called directly.
#             # otherwise, since we define `__getattr__` and redirect attribute
#             # query to `self.worker`, the method will be called on the worker.
#             return run_method(self, method, args, kwargs)
#         except Exception as e:
#             # if the driver worker also execute methods,
#             # exceptions in the rest worker may cause deadlock in rpc like ray
#             # see https://github.com/vllm-project/vllm/issues/3455
#             # print the error and inform the user to solve the error
#             msg = (f"Error executing method {method!r}. "
#                    "This might cause deadlock in distributed execution.")
#             logger.exception(msg)
#             raise e
        
#     def __getattr__(self, attr):
#         return getattr(self.worker, attr)


class Worker(ABC):

    def __init__(self, fastvideo_args: FastVideoArgs, local_rank: int,
                 rank: int, is_driver_worker: bool = False):
        self.fastvideo_args = fastvideo_args
        self.local_rank = local_rank
        self.rank = rank
        # TODO: don't hardcode this
        self.distributed_init_method = "env://"
        self.is_driver_worker = is_driver_worker

        context = zmq.Context(2)
        self.recv_from_rpc = get_zmq_socket(context, zmq.DEALER, "ipc://fastvideo_rpc_broadcast", False)
        # self.send_to_rpc = get_zmq_socket(context, zmq.PUSH, "inproc://fastvideo_rpc_broadcast", True)

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                # (TokenizedGenerateReqInput, self.handle_generate_request),
                # (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                # (FlushCacheReq, self.flush_cache_wrapped),
                # (AbortReq, self.abort_request),
                # (OpenSessionReqInput, self.open_session),
                # (CloseSessionReqInput, self.close_session),
                # (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                # (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                # (
                #     UpdateWeightsFromDistributedReqInput,
                #     self.update_weights_from_distributed,
                # ),
                # (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                # (GetWeightsByNameReqInput, self.get_weights_by_name),
                # (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                # (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                # (ProfileReq, self.profile),
                # (GetInternalStateReq, self.get_internal_state),
                # (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (GenerateRequest, self.handle_generate_request),
                # (ExpertDistributionReq, self.expert_distribution_handle),
            ]
        )
    
    def handle_rpc_request(self, req: RpcReqInput) -> RpcReqOutput:
        pass
    
    def handle_generate_request(self, req: GenerateRequest) -> None:
        logger.info(f"Worker {self.rank} received generate request")
        pass

    def init_device(self):
        if self.fastvideo_args.device_str.startswith("cuda"):
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
            raise ValueError(f"Unsupported device: {self.fastvideo_args.device_str}")

        # Initialize the distributed environment.
        init_worker_distributed_environment(self.fastvideo_args, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        
        self.pipeline = build_pipeline(self.fastvideo_args)

    def execute_forward(self, forward_batch: ForwardBatch) -> torch.Tensor:
        return self.pipeline.forward(forward_batch)
    
    def recv_requests(self) -> List[ForwardBatch]:
        pass

    def send_response(self, output: torch.Tensor) -> None:
        pass

    def event_loop(self) -> None:
        """Event loop for the worker."""
        logger.info(f"Worker {self.rank} starting event loop...")
        recv_rpc = self.recv_from_rpc.recv_pyobj()
        assert recv_rpc == "wait_until_ready"
        logger.info(f"Worker {self.rank} received wait_until_ready")
        self.recv_from_rpc.send_pyobj(f"ready{self.rank}")
        logger.info(f"Worker {self.rank} sent ready")
        logger.info(f"Worker {self.rank} started event loop")
        while True:
            logger.info(f"Worker {self.rank} waiting for RPC")
            recv_rpc = self.recv_from_rpc.recv_pyobj()
            logger.info(f"Received RPC: {recv_rpc}")
            # assert isinstance(recv_rpc, RpcReqInput)
            inputs = self._request_dispatcher(recv_rpc)
            output = self.execute_forward(inputs)
            self.send_response(output)
        

def init_worker_distributed_environment(
    fastvideo_args: FastVideoArgs,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize distributed environment and model parallelism."""

    world_size = fastvideo_args.num_gpus

    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 local_rank=local_rank)
    device_str = f"cuda:{local_rank}"
    fastvideo_args.device_str = device_str
    fastvideo_args.device = torch.device(device_str)
    assert fastvideo_args.sp_size is not None
    assert fastvideo_args.tp_size is not None
    initialize_model_parallel(
        sequence_model_parallel_size=fastvideo_args.sp_size,
        tensor_model_parallel_size=fastvideo_args.tp_size,
    )


def run_worker_process(fastvideo_args: FastVideoArgs, local_rank: int,
                 rank: int, pipe_writer: mp.Pipe):
    print(f"Worker {rank} starting...")
    try:
        # Config the process
        kill_itself_when_parent_died()
        prefix = f"{local_rank}"
        setproctitle.setproctitle(f"fastvideo::gpu_worker{prefix.replace(' ', '_')}")
        faulthandler.enable()
        parent_process = psutil.Process().parent()

        logger.info(f"Worker {rank} initializing...")
        worker = Worker(fastvideo_args, local_rank, rank, False)
        pipe_writer.send(
            {
                "status": "ready",
                "local_rank": local_rank,
            }
        )
        worker.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Worker {rank} hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
