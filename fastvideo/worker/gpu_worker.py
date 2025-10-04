# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod
import contextlib
import faulthandler
import multiprocessing as mp
import os
import signal
import sys
from multiprocessing.connection import Connection
from typing import Any, TextIO, cast

import psutil
import torch

import fastvideo.envs as envs
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel)
from fastvideo.distributed.parallel_state import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch, LoRAPipeline, build_pipeline
from fastvideo.platforms import current_platform
from fastvideo.utils import (get_exception_traceback,
                             kill_itself_when_parent_died,
                             run_method,
                             update_environment_variables)

logger = init_logger(__name__)

# ANSI color codes
CYAN = '\033[1;36m'
RESET = '\033[0;0m'


class GpuWorker:

    def __init__(self, fastvideo_args: FastVideoArgs, local_rank: int,
                 rank: int, pipe: Connection, master_port: int):
        self.fastvideo_args = fastvideo_args
        self.local_rank = local_rank
        self.rank = rank
        # TODO(will): don't hardcode this
        self.distributed_init_method = "env://"
        self.pipe = pipe
        self.master_port = master_port
        self.init_device()

        # Init request dispatcher
        # TODO(will): add request dispatcher: use TypeBasedDispatcher from
        # utils.py
        # self._request_dispatcher = TypeBasedDispatcher(
        #     [
        # (RpcReqInput, self.handle_rpc_request),
        # (GenerateRequest, self.handle_generate_request),
        # (ExpertDistributionReq, self.expert_distribution_handle),
        #     ]
        # )

    def init_device(self) -> None:
        """Initialize the device for the worker."""

        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

        # Platform-agnostic device initialization
        self.device = get_local_torch_device()

        # _check_if_gpu_supports_dtype(self.model_config.dtype)
        if current_platform.is_cuda_alike():
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            # For MPS, we can't get memory info the same way
            self.init_gpu_memory = 0

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.fastvideo_args.num_gpus)

        # Initialize the distributed environment.
        maybe_init_distributed_environment_and_model_parallel(
            self.fastvideo_args.tp_size, self.fastvideo_args.sp_size)

        self.pipeline = build_pipeline(self.fastvideo_args)

    def execute_forward(self, forward_batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        output_batch = self.pipeline.forward(forward_batch, self.fastvideo_args)
        return cast(ForwardBatch, output_batch)

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None) -> None:
        self.pipeline.set_lora_adapter(lora_nickname, lora_path)

    def shutdown(self) -> dict[str, Any]:
        """Gracefully shut down the worker process"""
        logger.info("Worker %d shutting down...",
                    self.rank,
                    local_main_process_only=False)
        # Clean up resources
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Clean up pipeline resources if needed
            pass

        # Destroy the distributed environment
        cleanup_dist_env_and_memory(shutdown_ray=False)

        logger.info("Worker %d shutdown complete",
                    self.rank,
                    local_main_process_only=False)
        return {"status": "shutdown_complete"}

    def unmerge_lora_weights(self) -> None:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.unmerge_lora_weights()

    def merge_lora_weights(self) -> None:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.merge_lora_weights()

    def event_loop(self) -> None:
        """Event loop for the worker."""
        logger.info("Worker %d starting event loop...",
                    self.rank,
                    local_main_process_only=False)
        while True:
            try:
                recv_rpc = self.pipe.recv()
                method_name = recv_rpc.get('method')

                # Handle shutdown request
                if method_name == 'shutdown':
                    response = self.shutdown()
                    with contextlib.suppress(Exception):
                        self.pipe.send(response)
                    break  # Exit the loop

                # Handle regular RPC calls
                if method_name == 'execute_forward':
                    forward_batch = recv_rpc['kwargs']['forward_batch']
                    fastvideo_args = recv_rpc['kwargs']['fastvideo_args']
                    output_batch = self.execute_forward(forward_batch,
                                                        fastvideo_args)
                    logging_info = None
                    if envs.FASTVIDEO_STAGE_LOGGING:
                        logging_info = output_batch.logging_info
                    self.pipe.send({
                        "output_batch": output_batch.output.cpu(),
                        "logging_info": logging_info
                    })
                elif method_name == 'set_lora_adapter':
                    lora_nickname = recv_rpc['kwargs']['lora_nickname']
                    lora_path = recv_rpc['kwargs']['lora_path']
                    self.set_lora_adapter(lora_nickname, lora_path)
                    logger.info("Worker %d set LoRA adapter %s with path %s",
                                self.rank, lora_nickname, lora_path)
                    self.pipe.send({"status": "lora_adapter_set"})
                elif method_name == 'unmerge_lora_weights':
                    self.unmerge_lora_weights()
                    logger.info("Worker %d unmerged LoRA weights", self.rank)
                    self.pipe.send({"status": "lora_adapter_unmerged"})
                elif method_name == 'merge_lora_weights':
                    self.merge_lora_weights()
                    logger.info("Worker %d merged LoRA weights", self.rank)
                    self.pipe.send({"status": "lora_adapter_merged"})
                else:
                    # Handle other methods dynamically if needed
                    args = recv_rpc.get('args', ())
                    kwargs = recv_rpc.get('kwargs', {})
                    if hasattr(self, method_name):
                        method = getattr(self, method_name)
                        result = method(*args, **kwargs)
                        self.pipe.send(result)
                    else:
                        self.pipe.send(
                            {"error": f"Unknown method: {method_name}"})
            except KeyboardInterrupt:
                logger.error(
                    "Worker %d in loop received KeyboardInterrupt, aborting forward pass",
                    self.rank)
                try:
                    self.pipe.send(
                        {"error": "Operation aborted by KeyboardInterrupt"})
                    logger.info("Worker %d sent error response after interrupt",
                                self.rank)
                except Exception as e:
                    logger.error("Worker %d failed to send error response: %s",
                                 self.rank, str(e))
                continue

class WorkerBase:
    """Worker interface that allows FastVideo to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        self.fastvideo_args = fastvideo_args
        self.current_platform = current_platform

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        raise NotImplementedError

    # def get_model(self) -> nn.Module:
        # raise NotImplementedError

    def load_model(self) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_forward(self, forward_batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        raise NotImplementedError

    def start_worker_execution_loop(self) -> None:
        """Execute model loop in parallel worker.

        You can stop the loop by executing a driver worker with an empty output.
        See `stop_remote_worker_execution_loop` for more details.
        """
        pass
        # with self.current_platform.inference_mode():
        #     while True:
        #         output = self.execute_forward()
        #         if output is None:
        #             return None

    def determine_num_available_blocks(self) -> tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        The implementation may run profiling or other heuristics to determine
        the size of caches.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

class LocalOrDistributedWorkerBase(WorkerBase):
    """
    Partial implementation of WorkerBase that has a default `execute_model`
    definition to perform metadata transfer between workers when in distributed
    mode. Subclasses of this interface should use model runners that inherit
    from ModelRunnerBase, and should only need to implement worker-local logic.
    If custom control plane logic is needed to transfer metadata, or if the
    model runner cannot inherit from ModelRunnerBase, use WorkerBase instead.
    """
    is_driver_worker: bool
    # model_runner: ModelRunnerBase
    # observability_config: Optional[ObservabilityConfig] = None

    @property
    @abstractmethod
    def do_metadata_broadcast(self) -> bool:
        """
        Used by the default `execute_model` to check whether broadcast is
        needed to transfer request inputs from the driver worker to other
        workers in the TP group. If WorkerBase subclass only supports
        single-worker execution, then this method should return False.
        """
        raise NotImplementedError


    # @abstractmethod
    # def prepare_worker_input(
    #         self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
    #     """
    #     Prepare the inputs to WorkerBase.execute_worker from an execution
    #     request. This method may move data to the worker's local device. It is
    #     not allowed to communicate with other workers or devices.
    #     """
    #     raise NotImplementedError

    # @abstractmethod
    # def execute_worker(self, worker_input: WorkerInput) -> None:
    #     """
    #     Process an execution request.
    #     """
    #     raise NotImplementedError

    # def _get_worker_input_from_broadcast(
    #     self
    # ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
    #         str, torch.Tensor]]]:
    #     """ Get the worker input from the broadcasted tensor dict. """
    #     assert self.do_metadata_broadcast
    #     assert not self.is_driver_worker
    #     broadcast_data = broadcast_tensor_dict(src=0)
    #     if not broadcast_data:
    #         return None
    #
    #     worker_input = WorkerInput.from_broadcasted_tensor_dict(broadcast_data)
    #     model_input = (
    #         self.model_runner.make_model_input_from_broadcasted_tensor_dict(
    #             broadcast_data))
    #
    #     kwargs = extract_previous_hidden_states(broadcast_data)
    #
    #     return model_input, worker_input, kwargs
    #
    # def _get_driver_input_and_broadcast(
    #     self, execute_model_req: ExecuteModelRequest
    # ) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]:
    #     """ Get the driver input and broadcast it to other workers.  """
    #     assert self.is_driver_worker
    #
    #     worker_input: WorkerInput = self.prepare_worker_input(
    #         execute_model_req=execute_model_req)
    #     model_input: ModelRunnerInputBase = (
    #         self.model_runner.prepare_model_input(
    #             execute_model_req.seq_group_metadata_list,
    #             execute_model_req.virtual_engine,
    #             execute_model_req.finished_requests_ids))
    #
    #     kwargs = extract_previous_hidden_states(execute_model_req)
    #
    #     if self.do_metadata_broadcast:
    #         broadcast_data = worker_input.as_broadcastable_tensor_dict()
    #         broadcast_data.update(model_input.as_broadcastable_tensor_dict())
    #         broadcast_data.update(kwargs)
    #         broadcast_tensor_dict(broadcast_data, src=0)
    #
    #     if execute_model_req.async_callback:
    #         model_input = dataclasses.replace(  # type: ignore
    #             model_input,
    #             async_callback=execute_model_req.async_callback)
    #
    #     return model_input, worker_input, kwargs
    #
    # def prepare_input(
    #     self,
    #     execute_model_req: Optional[ExecuteModelRequest] = None
    # ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
    #         str, torch.Tensor]]]:
    #     """
    #     Prepare the inputs to ModelRunner and workers.
    #     """
    #     if self.is_driver_worker:
    #         if execute_model_req is None:
    #             if self.do_metadata_broadcast:
    #                 # This signals that there's no more requests to process for
    #                 # now. All workers are running infinite loop with
    #                 # broadcast_tensor_dict, and it stops the loop when the
    #                 # driver broadcasts an empty input. Send an empty input to
    #                 # notify all other workers to stop their execution loop.
    #                 broadcast_tensor_dict({}, src=0)
    #             return None
    #         return self._get_driver_input_and_broadcast(execute_model_req)
    #     else:
    #         return self._get_worker_input_from_broadcast()
    #
    # def get_model(self) -> nn.Module:
    #     return self.model_runner.get_model()
    #
    # def execute_model(
    #     self,
    #     execute_model_req: Optional[ExecuteModelRequest] = None,
    # ) -> Optional[List[SamplerOutput]]:
    #     """Executes at least one model step on the given sequences, unless no
    #     sequences are provided."""
    #     start_time = time.perf_counter()
    #
    #     inputs = self.prepare_input(execute_model_req)
    #     if inputs is None:
    #         return None
    #
    #     model_input, worker_input, kwargs = inputs
    #     num_steps = worker_input.num_steps
    #
    #     self.execute_worker(worker_input)
    #
    #     # If there is no input, we don't need to execute the model.
    #     if worker_input.num_seq_groups == 0:
    #         return []
    #
    #     intermediate_tensors = None
    #     orig_model_execute_time = 0.0
    #     if not get_pp_group().is_first_rank:
    #         intermediate_tensors = IntermediateTensors(
    #             get_pp_group().recv_tensor_dict(
    #                 all_gather_group=get_tp_group()))
    #         if (self.observability_config is not None
    #                 and self.observability_config.collect_model_execute_time):
    #             orig_model_execute_time = intermediate_tensors.tensors.get(
    #                 "model_execute_time", torch.tensor(0)).item()
    #
    #     output = self.model_runner.execute_model(
    #         model_input=model_input,
    #         kv_caches=self.kv_cache[worker_input.virtual_engine]
    #         if self.kv_cache is not None else None,
    #         intermediate_tensors=intermediate_tensors,
    #         num_steps=num_steps,
    #         **kwargs,
    #     )
    #
    #     model_execute_time = time.perf_counter() - start_time
    #     if not get_pp_group().is_last_rank:
    #         # output is IntermediateTensors
    #         assert isinstance(output, IntermediateTensors)
    #         if (self.observability_config is not None
    #                 and self.observability_config.collect_model_execute_time):
    #             output.tensors["model_execute_time"] = torch.tensor(
    #                 model_execute_time + orig_model_execute_time)
    #         get_pp_group().send_tensor_dict(output.tensors,
    #                                         all_gather_group=get_tp_group())
    #         return [None]
    #     if (self.observability_config is not None
    #             and self.observability_config.collect_model_execute_time
    #             and output is not None):
    #         for o in output:
    #             o.model_execute_time = (orig_model_execute_time +
    #                                     model_execute_time)
    #
    #     # output is List[SamplerOutput]
    #     return output

    # def _execute_model_spmd(
    #     self,
    #     execute_model_req: ExecuteModelRequest,
    #     intermediate_tensors: Optional[IntermediateTensors] = None
    # ) -> Optional[List[SamplerOutput]]:
    #     """
    #     Execute model in Single Program Multiple Data (SPMD) fashion.
    #     All workers take the same request, prepare the input and
    #     execute the model.
    #     """
    #     assert execute_model_req is not None, (
    #         "_execute_model_spmd() requires each worker to take in an "
    #         "ExecuteModelRequest")
    #     worker_input: WorkerInput = self.prepare_worker_input(
    #         execute_model_req=execute_model_req)
    #     model_input: ModelRunnerInputBase = (
    #         self.model_runner.prepare_model_input(
    #             execute_model_req.seq_group_metadata_list))
    #
    #     self.execute_worker(worker_input)
    #
    #     # If there is no input, we don't need to execute the model.
    #     if worker_input.num_seq_groups == 0:
    #         return []
    #
    #     kwargs = extract_previous_hidden_states(execute_model_req)
    #
    #     return self.model_runner.execute_model(
    #         model_input=model_input,
    #         kv_caches=self.kv_cache[worker_input.virtual_engine]
    #         if self.kv_cache is not None else None,
    #         intermediate_tensors=intermediate_tensors,
    #         **kwargs,
    #     )

class Worker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        fastvideo_args: FastVideoArgs,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        WorkerBase.__init__(self, fastvideo_args)
        # self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

    # def start_profile(self):
    #     if self.profiler is None:
    #         raise RuntimeError("Profiler is not enabled.")
    #     self.profiler.start()
    #
    # def stop_profile(self):
    #     if self.profiler is None:
    #         raise RuntimeError("Profiler is not enabled.")
    #     self.profiler.stop()
    #     print(
    #         self.profiler.key_averages().table(sort_by="self_cuda_time_total"))

    # def sleep(self, level: int = 1) -> None:
    #     free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
    #
    #     # Save the buffers before level 2 sleep
    #     if level == 2:
    #         model = self.model_runner.model
    #         self._sleep_saved_buffers = {
    #             name: buffer.cpu().clone()
    #             for name, buffer in model.named_buffers()
    #         }
    #
    #     allocator = CuMemAllocator.get_instance()
    #     allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
    #     free_bytes_after_sleep, total = torch.cuda.mem_get_info()
    #     freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
    #     used_bytes = total - free_bytes_after_sleep
    #     assert freed_bytes >= 0, "Memory usage increased after sleeping."
    #     logger.info(
    #         "Sleep mode freed %.2f GiB memory, "
    #         "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
    #         used_bytes / GiB_bytes)
    #
    # def wake_up(self, tags: Optional[list[str]] = None) -> None:
    #     allocator = CuMemAllocator.get_instance()
    #     allocator.wake_up(tags=tags)
    #
    #     # Restore the buffers after level 2 sleep
    #     if len(self._sleep_saved_buffers):
    #         model = self.model_runner.model
    #         for name, buffer in model.named_buffers():
    #             if name in self._sleep_saved_buffers:
    #                 buffer.data.copy_(self._sleep_saved_buffers[name].data)
    #         self._sleep_saved_buffers = {}

    def init_device(self) -> None:
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

        # Platform-agnostic device initialization
        self.device = get_local_torch_device()
        logger.info(f"xxx-os.environ={os.environ["CUDA_VISIBLE_DEVICES"]}")

        # _check_if_gpu_supports_dtype(self.model_config.dtype)
        # if current_platform.is_cuda_alike():
        #     self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        # else:
        #     # For MPS, we can't get memory info the same way
        #     self.init_gpu_memory = 0

        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = str(self.master_port)

        # in ray cluster, we shouldn't pass local_rank to decide device name
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.fastvideo_args.num_gpus)

        # Initialize the distributed environment.
        maybe_init_distributed_environment_and_model_parallel(
            self.fastvideo_args.tp_size, self.fastvideo_args.sp_size, self.distributed_init_method)

        self.pipeline = build_pipeline(self.fastvideo_args)

    # def load_model(self):
    #     if self.vllm_config.model_config.enable_sleep_mode:
    #         allocator = CuMemAllocator.get_instance()
    #         assert allocator.get_current_usage() == 0, (
    #             "Sleep mode can only be "
    #             "used for one instance per process.")
    #         context = allocator.use_memory_pool(tag="weights")
    #     else:
    #         context = nullcontext()
    #     with context:
    #         self.model_runner.load_model()

    # def save_sharded_state(
    #     self,
    #     path: str,
    #     pattern: Optional[str] = None,
    #     max_size: Optional[int] = None,
    # ) -> None:
    #     self.model_runner.save_sharded_state(
    #         path,
    #         pattern=pattern,
    #         max_size=max_size,
    #     )

    # def save_tensorized_model(
    #     self,
    #     tensorizer_config: TensorizerConfig,
    # ) -> None:
    #     self.model_runner.save_tensorized_model(
    #         tensorizer_config=tensorizer_config, )

    # @torch.inference_mode()
    # def determine_num_available_blocks(self) -> Tuple[int, int]:
    #     """Profiles the peak memory usage of the model to determine how many
    #     KV blocks may be allocated without OOMs.
    #
    #     The engine will first conduct a profiling of the existing memory usage.
    #     Then, it calculate the maximum possible number of GPU and CPU blocks
    #     that can be allocated with the remaining free memory.
    #
    #     Tip:
    #         You may limit the usage of GPU memory
    #         by adjusting the `gpu_memory_utilization` parameter.
    #     """
    #     # Profile the memory usage of the model and get the maximum number of
    #     # cache blocks that can be allocated with the remaining free memory.
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()
    #
    #     free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()
    #
    #     # Execute a forward pass with dummy inputs to profile the memory usage
    #     # of the model.
    #     with memory_profiling(
    #             self.baseline_snapshot,
    #             weights_memory=self.model_runner.model_memory_usage) as result:
    #         self.model_runner.profile_run()
    #
    #     self._assert_memory_footprint_increased_during_profiling()
    #
    #     memory_for_current_instance = total_gpu_memory * \
    #         self.cache_config.gpu_memory_utilization
    #     available_kv_cache_memory = (memory_for_current_instance -
    #                                  result.non_kv_cache_memory)
    #
    #     # Calculate the number of blocks that can be allocated with the
    #     # profiled peak memory.
    #     cache_block_size = self.get_cache_block_size_bytes()
    #     if cache_block_size == 0:
    #         num_gpu_blocks = 0
    #         num_cpu_blocks = 0
    #     else:
    #         num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
    #         num_cpu_blocks = int(self.cache_config.swap_space_bytes //
    #                              cache_block_size)
    #     num_gpu_blocks = max(num_gpu_blocks, 0)
    #     num_cpu_blocks = max(num_cpu_blocks, 0)
    #
    #     msg = (f"Memory profiling takes {result.profile_time:.2f} seconds\n"
    #            "the current vLLM instance can use "
    #            "total_gpu_memory "
    #            f"({(total_gpu_memory / GiB_bytes):.2f}GiB)"
    #            " x gpu_memory_utilization "
    #            f"({self.cache_config.gpu_memory_utilization:.2f})"
    #            f" = {(memory_for_current_instance / GiB_bytes):.2f}GiB\n"
    #            "model weights take "
    #            f"{(result.weights_memory / GiB_bytes):.2f}GiB;"
    #            " non_torch_memory takes "
    #            f"{(result.non_torch_increase / GiB_bytes):.2f}GiB;"
    #            " PyTorch activation peak memory takes "
    #            f"{(result.torch_peak_increase / GiB_bytes):.2f}GiB;"
    #            " the rest of the memory reserved for KV Cache is "
    #            f"{(available_kv_cache_memory / GiB_bytes):.2f}GiB.")
    #
    #     logger.info(msg)
    #     # Final cleanup
    #     gc.collect()
    #
    #     return num_gpu_blocks, num_cpu_blocks

    # def _assert_memory_footprint_increased_during_profiling(self):
    #     # NOTE(woosuk): Here we assume that the other processes using the same
    #     # GPU did not change their memory usage during the profiling.
    #     free_gpu_memory, total = torch.cuda.mem_get_info()
    #     cuda_memory = total - free_gpu_memory
    #     assert self.baseline_snapshot.cuda_memory < cuda_memory, (
    #         "Error in memory profiling. "
    #         f"Initial used memory {self.baseline_snapshot.cuda_memory}, "
    #         f"currently used memory {cuda_memory}. "
    #         f"This happens when the GPU memory was "
    #         "not properly cleaned up before initializing the vLLM instance.")

    # def initialize_cache(self, num_gpu_blocks: int,
    #                      num_cpu_blocks: int) -> None:
    #     """Allocate GPU and CPU KV cache with the specified number of blocks.
    #
    #     This also warms up the model, which may record CUDA graphs.
    #     """
    #     raise_if_cache_size_invalid(
    #         num_gpu_blocks, self.cache_config.block_size,
    #         self.cache_config.is_attention_free,
    #         self.model_config.max_model_len,
    #         self.parallel_config.pipeline_parallel_size)
    #
    #     self.cache_config.num_gpu_blocks = num_gpu_blocks
    #     self.cache_config.num_cpu_blocks = num_cpu_blocks
    #
    #     if self.vllm_config.model_config.enable_sleep_mode:
    #         allocator = CuMemAllocator.get_instance()
    #         context = allocator.use_memory_pool(tag="kv_cache")
    #     else:
    #         context = nullcontext()
    #     with context:
    #         self._init_cache_engine()
    #     self._warm_up_model()

    # def _init_cache_engine(self):
    #     assert self.cache_config.num_gpu_blocks is not None
    #     self.cache_engine = [
    #         CacheEngine(self.cache_config, self.model_config,
    #                     self.parallel_config, self.device_config)
    #         for _ in range(self.parallel_config.pipeline_parallel_size)
    #     ]
    #     self.gpu_cache = [
    #         self.cache_engine[ve].gpu_cache
    #         for ve in range(self.parallel_config.pipeline_parallel_size)
    #     ]
    #
    #     # Layer pairings for cross-layer KV sharing.
    #     # If an Attention layer `layer_name` is in the keys of this dict, it
    #     # means this layer will perform attention using the keys and values
    #     # from the KV cache of `shared_kv_cache_layers[layer_name]`.
    #     shared_kv_cache_layers: dict[str, str] = {}
    #
    #     attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
    #
    #     for layer_name, attn_module in attn_layers.items():
    #         if (kv_tgt_layer :=
    #                 attn_module.kv_sharing_target_layer_name) is not None:
    #             # The layer doesn't need its own KV cache and will use that of
    #             # the target layer. We skip creating a KVCacheSpec for it, so
    #             # that KV cache management logic will act as this layer does
    #             # not exist, and doesn't allocate KV cache for the layer. This
    #             # enables the memory saving of cross-layer kv sharing, allowing
    #             # a given amount of memory to accommodate longer context lengths
    #             # or enable more requests to be processed simultaneously.
    #             shared_kv_cache_layers[layer_name] = kv_tgt_layer
    #
    #     bind_kv_cache(self.compilation_config.static_forward_context,
                      # self.gpu_cache, shared_kv_cache_layers)

    # def _warm_up_model(self) -> None:
    #     # warm up sizes that are not in cudagraph capture sizes,
    #     # but users still want to compile for better performance,
    #     # e.g. for the max-num-batched token size in chunked prefill.
    #     warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
    #     if not self.model_config.enforce_eager:
    #         warmup_sizes = [
    #             x for x in warmup_sizes if x not in
    #             self.vllm_config.compilation_config.cudagraph_capture_sizes
    #         ]
    #     for size in sorted(warmup_sizes, reverse=True):
    #         logger.info("Compile and warming up model for size %d", size)
    #         self.model_runner._dummy_run(size)
    #     if not self.model_config.enforce_eager:
    #         self.model_runner.capture_model(self.gpu_cache)
    #     # Reset the seed to ensure that the random state is not affected by
    #     # the model initialization and profiling.
    #     set_random_seed(self.model_config.seed)

    # @property
    # def do_metadata_broadcast(self) -> bool:
    #     return self.parallel_config.tensor_parallel_size > 1

    # @property
    # def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
    #     return self.gpu_cache

    # @torch.inference_mode()
    # def prepare_worker_input(
    #         self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
    #     virtual_engine = execute_model_req.virtual_engine
    #     num_steps = execute_model_req.num_steps
    #     num_seq_groups = len(execute_model_req.seq_group_metadata_list)
    #     # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
    #     # they contain parameters to launch cudamemcpyasync.
    #     blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
    #                                      device="cpu",
    #                                      dtype=torch.int64).view(-1, 2)
    #     blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
    #                                       device="cpu",
    #                                       dtype=torch.int64).view(-1, 2)
    #     # `blocks_to_copy` is a gpu tensor. The src and tgt of
    #     # blocks to copy are in the same device, and `blocks_to_copy`
    #     # can be used directly within cuda kernels.
    #     blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
    #                                   device=self.device,
    #                                   dtype=torch.int64).view(-1, 2)
    #
    #     return WorkerInput(
    #         num_seq_groups=num_seq_groups,
    #         blocks_to_swap_in=blocks_to_swap_in,
    #         blocks_to_swap_out=blocks_to_swap_out,
    #         blocks_to_copy=blocks_to_copy,
    #         virtual_engine=virtual_engine,
    #         num_steps=num_steps,
    #     )

    # @torch.inference_mode()
    # def execute_worker(self, worker_input: WorkerInput) -> None:
    #     virtual_engine = worker_input.virtual_engine
    #     # Issue cache operations.
    #     if (worker_input.blocks_to_swap_in is not None
    #             and worker_input.blocks_to_swap_in.numel() > 0):
    #         self.cache_engine[virtual_engine].swap_in(
    #             worker_input.blocks_to_swap_in)
    #     if (worker_input.blocks_to_swap_out is not None
    #             and worker_input.blocks_to_swap_out.numel() > 0):
    #         self.cache_engine[virtual_engine].swap_out(
    #             worker_input.blocks_to_swap_out)
    #     if (worker_input.blocks_to_copy is not None
    #             and worker_input.blocks_to_copy.numel() > 0):
    #         self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

    # def _get_cached_seq_group_metadata(
    #         self,
    #         seq_group_metadata_list: List[Union[SequenceGroupMetadata,
    #                                             SequenceGroupMetadataDelta]],
    #         finished_request_ids: List[str]) -> List[SequenceGroupMetadata]:
    #     """Return a list of cached Sequence Group Metadata after updating its
    #     state.
    #
    #     It is used because scheduler only sends delta to workers to reduce
    #     the data payload size. The function also cleans up cache based on
    #     a given `finished_request_ids`.
    #     """
    #     new_seq_group_metadata_list = []
    #     for metadata_or_delta in seq_group_metadata_list:
    #         request_id = metadata_or_delta.request_id
    #         if request_id not in self._seq_group_metadata_cache:
    #             # The first prefill.
    #             assert isinstance(metadata_or_delta, SequenceGroupMetadata)
    #             self._seq_group_metadata_cache[request_id] = metadata_or_delta
    #         else:
    #             # The first prefill is already cached.
    #             if isinstance(metadata_or_delta, SequenceGroupMetadataDelta):
    #                 self._seq_group_metadata_cache[request_id].apply_delta(
    #                     metadata_or_delta)
    #             else:
    #                 # If metadata snapshot is sent again, it is
    #                 # preempted. Reset the cache because we need to start
    #                 # from scratch.
    #                 assert isinstance(metadata_or_delta, SequenceGroupMetadata)
    #                 self._seq_group_metadata_cache[
    #                     request_id] = metadata_or_delta
    #
    #         new_seq_group_metadata_list.append(
    #             self._seq_group_metadata_cache[request_id])
    #
    #     # Clean up finished ids
    #     for finished_id in finished_request_ids:
    #         del self._seq_group_metadata_cache[finished_id]
    #
    #     return new_seq_group_metadata_list

    # def _execute_model_spmd(
    #     self,
    #     execute_model_req: ExecuteModelRequest,
    #     intermediate_tensors: Optional[IntermediateTensors] = None,
    # ) -> Optional[List[SamplerOutput]]:
    #     if execute_model_req is not None:
    #         new_seq_group_metadata_list = self._get_cached_seq_group_metadata(
    #             execute_model_req.seq_group_metadata_list,
    #             execute_model_req.finished_requests_ids)
    #
    #         execute_model_req.seq_group_metadata_list = (
    #             new_seq_group_metadata_list)
    #     output = super()._execute_model_spmd(execute_model_req,
    #                                          intermediate_tensors)
    #     return output

    def execute_forward(self, forward_batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        output_batch = self.pipeline.forward(forward_batch, self.fastvideo_args)
        return cast(ForwardBatch, output_batch)

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None) -> None:
        self.pipeline.set_lora_adapter(lora_nickname, lora_path)

    def unmerge_lora_weights(self) -> None:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.unmerge_lora_weights()

    def merge_lora_weights(self) -> None:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.merge_lora_weights()


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        fastvideo_args: FastVideoArgs,
        rpc_rank: int = 0,
    ) -> None:
        """
        Initialize the worker wrapper with the given fastvideo_args and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank = rpc_rank
        self.worker: WorkerBase | None = None
        self.fastvideo_args: FastVideoArgs | None = None
        # do not store this `fastvideo_args`, `init_worker` will set the final
        # one.

    def adjust_rank(self, rank_mapping: dict[int, int]) -> None:
        """
        Adjust the rpc_rank based on the given mapping.
        It is only used during the initialization of the executor,
        to adjust the rpc_rank of workers after we create all workers.
        """
        if self.rpc_rank in rank_mapping:
            self.rpc_rank = rank_mapping[self.rpc_rank]

    def update_environment_variables(self, envs_list: list[dict[str,
                                                                str]]) -> None:
        envs = envs_list[self.rpc_rank]
        key = 'CUDA_VISIBLE_DEVICES'
        logger.info(f"xxx-before-envs={envs}, os.environ={os.environ[key]}")
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)
        logger.info(f"xxx-after-envs={envs}, os.environ={os.environ[key]}")

    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        # TODO(xingyu): move this to RayWorkerWrapper
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]
        self.fastvideo_args = kwargs.get("fastvideo_args")
        assert self.fastvideo_args is not None, (
            "fastvideo_args is required to initialize the worker")
        # enable_trace_function_call_for_thread(self.vllm_config)

        # from vllm.plugins import load_general_plugins
        # load_general_plugins()

        # TODO(xingyu): we only support ray at WorkerWrapperBase
        # if self.fastvideo_args.distributed_executor_backend == "ray":
        #     worker_class = None
        #
        # # To make FastVideo args available during worker initialization
        self.worker = Worker(**kwargs)
        assert self.worker is not None

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.rpc_rank]
        self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        # To make FastVideo args available during device initialization
        self.worker.init_device()  # type: ignore

    def execute_method(self, method: str | bytes, *args, **kwargs):
        try:
            # method resolution order:
            # if a method is defined in this class, it will be called directly.
            # otherwise, since we define `__getattr__` and redirect attribute
            # query to `self.worker`, the method will be called on the worker.
            return run_method(self, method, args, kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method!r}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e

    def __getattr__(self, attr):
        return getattr(self.worker, attr)


def run_worker_process(fastvideo_args: FastVideoArgs, local_rank: int,
                       rank: int, pipe: Connection, master_port: int):
    # Add process-specific prefix to stdout and stderr
    process_name = mp.current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    # Config the process
    kill_itself_when_parent_died()
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    logger.info("Worker %d initializing...",
                rank,
                local_main_process_only=False)

    try:
        worker = GpuWorker(fastvideo_args, local_rank, rank, pipe, master_port)
        logger.info("Worker %d sending ready", rank)
        pipe.send({
            "status": "ready",
            "local_rank": local_rank,
        })
        worker.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("Worker %d hit an exception: %s", rank, traceback)
        parent_process.send_signal(signal.SIGQUIT)


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Prepend each output line with process-specific prefix"""

    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find('\n', idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]
