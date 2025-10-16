# SPDX-License-Identifier: Apache-2.0
# Adapt from https://github.com/vllm-project/vllm/blob/releases/v0.11.0/vllm/executor/ray_distributed_executor.py

from collections import defaultdict
import os
import cloudpickle

import fastvideo.envs as envs
from dataclasses import dataclass

from typing import Any, TYPE_CHECKING
from collections.abc import Callable
from fastvideo.utils import get_ip, get_distributed_init_method, get_open_port
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.worker.executor import Executor
from fastvideo.worker.ray_utils import (
    initialize_ray_cluster,
    RayWorkerWrapper,
    ray,
)
from fastvideo.worker.ray_env import get_env_vars_to_copy
from fastvideo.logger import init_logger

if ray is not None:
    from ray.actor import ActorHandle
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)


@dataclass
class RayWorkerMetaData:
    """
    Metadata for a Ray worker.
    The order of ray worker creation can be random,
    and we need to reset the rank after creating all workers.
    """

    worker: ActorHandle
    created_rank: int
    adjusted_rank: int = -1
    ip: str = ""


class RayDistributedExecutor(Executor):
    """Ray-based distributed executor"""

    # These env vars are worker-specific, therefore are NOT copied
    # from the driver to the workers
    WORKER_SPECIFIC_ENV_VARS = {
        "FASTVIDEO_HOST_IP",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
    }

    # These non-vLLM env vars are copied from the driver to workers
    ADDITIONAL_ENV_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}

    def _init_executor(self) -> None:
        initialize_ray_cluster(self.fastvideo_args)
        placement_group = self.fastvideo_args.ray_placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        self._init_workers_ray(placement_group)

    # child class could overwrite this to return actual env vars.
    def _get_env_vars_to_be_updated(self) -> list[dict[str, str]]:
        return self._env_vars_for_all_workers

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        from fastvideo.platforms import current_platform

        num_gpus = envs.FASTVIDEO_RAY_PER_WORKER_GPUS

        # The remaining workers are the actual ray actors.
        self.workers: list[RayWorkerWrapper] = []

        # Create the workers.
        # use the first N bundles that have GPU resources.
        bundle_indices: list[int] = []
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if bundle.get(current_platform.ray_device_key, 0):
                bundle_indices.append(bundle_id)
        bundle_indices = bundle_indices[:self.fastvideo_args.num_gpus]

        worker_metadata: list[RayWorkerMetaData] = []
        driver_ip = get_ip()
        for rank, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if current_platform.ray_device_key == "GPU":
                # NV+AMD GPUs, and Intel XPUs
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(fastvideo_args=self.fastvideo_args,
                                           rpc_rank=rank)
            else:
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_gpus},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(fastvideo_args=self.fastvideo_args,
                                           rpc_rank=rank)
            worker_metadata.append(
                RayWorkerMetaData(worker=worker, created_rank=rank))

        worker_ips = ray.get([
            each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
            for each in worker_metadata
        ])

        for each, ip in zip(worker_metadata, worker_ips, strict=False):
            each.ip = ip

        logger.debug("workers: %s", worker_metadata)

        ip_counts: dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item.ip
            return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        sorted_worker_metadata = sorted(worker_metadata,
                                        key=sort_by_driver_then_worker_ip)
        start_rank = 0
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i + start_rank
        self.workers = [item.worker for item in sorted_worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank
            for item in sorted_worker_metadata
        }
        self._run_ray_workers("adjust_rank", rerank_mapping)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_ray_workers("get_node_and_gpu_ids")

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            # `gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `FASTVIDEO_HOST_IP`"
                " environment variable, make sure it is unique for"
                " each node.")

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables: list[dict[str, str]] = [{
            current_platform.device_control_env_var:
            ",".join(map(str, node_gpus[node_id])),
        } for (node_id, _) in worker_node_and_gpu_ids]

        # Environment variables to copy from driver to workers
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=self.WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars).union(
                self.ADDITIONAL_ENV_VARS),
            destination="workers",
        )

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers: list[dict[str, str]] = (
            all_args_to_update_environment_variables)

        self._run_ray_workers("update_environment_variables",
                              self._get_env_vars_to_be_updated())

        if len(node_gpus) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = node_workers[node_id].index(rank)
            kwargs = dict(
                fastvideo_args=self.fastvideo_args,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
            )
            all_kwargs.append(kwargs)
        self._run_ray_workers("init_worker", all_kwargs)
        self._run_ray_workers("init_device")

        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: list[RayWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: list[RayWorkerWrapper] = []

        # Enforce rank order for correct rank to return final output.
        for index, worker in enumerate(self.workers):
            # The driver worker is rank 0 and not in self.workers.
            rank = index + 1
            if rank % self.fastvideo_args.tp_size == 0:
                self.tp_driver_workers.append(worker)
            else:
                self.non_driver_workers.append(worker)

    def execute_forward(self, forward_batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        responses: list[ForwardBatch] = self.collective_rpc(
            "execute_forward",
            kwargs={
                "forward_batch": forward_batch,
                "fastvideo_args": fastvideo_args,
            },
        )
        output = responses[0].output.cpu()

        logging_info = None
        if envs.FASTVIDEO_STAGE_LOGGING:
            logging_info = responses[0].logging_info

        result_batch = ForwardBatch(
            data_type=forward_batch.data_type,
            output=output,
            logging_info=logging_info,
        )
        return result_batch

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None) -> None:
        responses = self.collective_rpc("set_lora_adapter",
                                        kwargs={
                                            "lora_nickname": lora_nickname,
                                            "lora_path": lora_path
                                        })
        for i, response in enumerate(responses):
            if response["status"] != "lora_adapter_set":
                raise RuntimeError(
                    f"Worker {i} failed to set LoRA adapter to {lora_path}")

    def unmerge_lora_weights(self) -> None:
        responses = self.collective_rpc("unmerge_lora_weights", kwargs={})
        for i, response in enumerate(responses):
            if response["status"] != "lora_adapter_unmerged":
                raise RuntimeError(f"Worker {i} failed to unmerge LoRA weights")

    def merge_lora_weights(self) -> None:
        responses = self.collective_rpc("merge_lora_weights", kwargs={})
        for i, response in enumerate(responses):
            if response["status"] != "lora_adapter_merged":
                raise RuntimeError(f"Worker {i} failed to merge LoRA weights")

    def collective_rpc(self,
                       method: str | Callable,
                       timeout: float | None = None,
                       args: tuple = (),
                       kwargs: dict | None = None) -> list[Any]:
        return self._run_ray_workers(method, *args, **(kwargs or {}))

    def _run_ray_workers(
        self,
        method: str | Callable,
        *args,
        **kwargs,
    ) -> Any:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        # Start the ray workers first.
        ray_workers = self.workers
        ray_worker_outputs = [
            worker.execute_method.remote(sent_method, *args, **kwargs)
            for worker in ray_workers
        ]

        # Get the results of the ray workers.
        ray_worker_outputs = ray.get(ray_worker_outputs)

        return ray_worker_outputs

    def shutdown(self) -> None:
        logger.info(
            "Shutting down Ray distributed executor. If you see error log "
            "from logging.cc regarding SIGTERM received, please ignore because "
            "this is the expected termination process in Ray.")
        import ray
        for worker in self.workers:
            ray.kill(worker)

        self.workers = []

    def __del__(self):
        self.shutdown()
