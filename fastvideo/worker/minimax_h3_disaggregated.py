# SPDX-License-Identifier: Apache-2.0
"""Ray data-plane for component-disaggregated MiniMax H3 inference."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import suppress
from copy import deepcopy
import os
from queue import Queue
from typing import Any, cast
from uuid import uuid4

import torch

from fastvideo.distributed import cleanup_dist_env_and_memory
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.minimax_h3.disaggregated import (
    MiniMaxH3DenoisedState,
    MiniMaxH3DiTPipeline,
    MiniMaxH3EncodedState,
    MiniMaxH3EncoderDecoderPipeline,
    MiniMaxH3RefDiTPipeline,
    MiniMaxH3RefEncoderDecoderPipeline,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.lazy_module import is_lazy_module
from fastvideo.utils import get_ip, get_open_port
from fastvideo.worker.executor import Executor
from fastvideo.worker.ray_utils import assert_ray_available, ray

logger = init_logger(__name__)


def _node_resource(node_ip: str) -> str:
    return f"node:{node_ip}"


def _validate_topology(encoder_node_ip: str, dit_node_ip: str, resources: dict[str, float]) -> None:
    if not encoder_node_ip or not dit_node_ip:
        raise ValueError("MiniMax-H3 disaggregation requires both encoder and DiT node IPs.")
    if encoder_node_ip == dit_node_ip:
        raise ValueError("MiniMax-H3 encoder/decoder and DiT workers must use different Ray nodes.")
    missing = [ip for ip in (encoder_node_ip, dit_node_ip) if resources.get(_node_resource(ip), 0.0) <= 0]
    if missing:
        available = sorted(key.removeprefix("node:") for key in resources if key.startswith("node:"))
        raise RuntimeError(f"Ray has no live node resource for {missing}; available node IPs: {available}.")


def _resident_role_args(source: FastVideoArgs, *, role: str) -> FastVideoArgs:
    """Clone process-local args and force one persistent, non-parallel component role."""
    args = deepcopy(source)
    args.num_gpus = 1
    args.tp_size = 1
    args.sp_size = 1
    args.hsdp_replicate_dim = 1
    args.hsdp_shard_dim = 1
    args.ray_placement_group = None
    args.ray_runtime_env = None
    args.distributed_executor_backend = "mp"
    args.use_fsdp_inference = False
    args.dit_cpu_offload = False
    args.dit_layerwise_offload = False
    args.text_encoder_cpu_offload = False
    args.image_encoder_cpu_offload = False
    args.vae_cpu_offload = False
    args.lazy_module_load = False
    args.h3_sequential_load = False
    args.vae_parallel_encode = False
    args.vae_parallel_decode = False
    if role == "encoder_decoder":
        # Adapters target the transformer and belong exclusively on Spark B.
        args.lora_path = None
        args.enable_torch_compile = False
    elif role == "dit":
        args.enable_torch_compile_text_encoder = False
        args.enable_torch_compile_vae = False
        args.enable_torch_compile_audio_vae = False
    else:
        raise ValueError(f"Unknown MiniMax-H3 worker role: {role!r}.")
    return args


def _bind_single_gpu_process() -> None:
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(get_open_port())
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


def _is_ref2va(args: FastVideoArgs) -> bool:
    return args.override_pipeline_cls_name == "MiniMaxH3Ref2VAModularPipeline"


def _request_id(batch: ForwardBatch) -> str:
    supplied = batch.extra.get("request_id")
    return str(supplied) if supplied is not None else uuid4().hex


def _next_or_sentinel(iterator: Iterator[ForwardBatch], sentinel: object) -> ForwardBatch | object:
    return next(iterator, sentinel)


class _MiniMaxH3EncoderDecoderActor:

    def __init__(self, fastvideo_args: FastVideoArgs) -> None:
        _bind_single_gpu_process()
        args = _resident_role_args(fastvideo_args, role="encoder_decoder")
        pipeline_cls = MiniMaxH3RefEncoderDecoderPipeline if _is_ref2va(args) else MiniMaxH3EncoderDecoderPipeline
        self.pipeline = pipeline_cls(args.model_path, args)
        self.pipeline.post_init()

    def encode(self, batch: ForwardBatch, request_id: str) -> MiniMaxH3EncodedState:
        return self.pipeline.encode(batch, request_id=request_id)

    def decode(self, state: MiniMaxH3DenoisedState) -> ForwardBatch:
        return self.pipeline.decode(state)

    def health(self) -> dict[str, Any]:
        modules = tuple(sorted(self.pipeline.modules))
        return {
            "ready": True,
            "role": "encoder_decoder",
            "node_ip": get_ip(),
            "modules": modules,
            "all_resident": all(not is_lazy_module(module) for module in self.pipeline.modules.values()),
        }

    def shutdown(self) -> dict[str, str]:
        self.pipeline = None
        cleanup_dist_env_and_memory(shutdown_ray=False)
        return {"status": "shutdown_complete"}


class _MiniMaxH3DiTActor:

    def __init__(self, fastvideo_args: FastVideoArgs) -> None:
        _bind_single_gpu_process()
        args = _resident_role_args(fastvideo_args, role="dit")
        pipeline_cls = MiniMaxH3RefDiTPipeline if _is_ref2va(args) else MiniMaxH3DiTPipeline
        self.pipeline = pipeline_cls(args.model_path, args)
        self.pipeline.post_init()

    def denoise(self, state: MiniMaxH3EncodedState) -> MiniMaxH3DenoisedState:
        return self.pipeline.denoise(state)

    def set_lora_adapter(self, lora_nickname: str, lora_path: str | None, strength: float,
                         accumulate: bool) -> dict[str, str]:
        self.pipeline.set_lora_adapter(lora_nickname, lora_path, strength=strength, accumulate=accumulate)
        return {"status": "lora_adapter_set"}

    def unmerge_lora_weights(self) -> dict[str, str]:
        self.pipeline.unmerge_lora_weights()
        return {"status": "lora_adapter_unmerged"}

    def merge_lora_weights(self) -> dict[str, str]:
        self.pipeline.merge_lora_weights()
        return {"status": "lora_adapter_merged"}

    def health(self) -> dict[str, Any]:
        modules = tuple(sorted(self.pipeline.modules))
        return {
            "ready": True,
            "role": "dit",
            "node_ip": get_ip(),
            "modules": modules,
            "all_resident": all(not is_lazy_module(module) for module in self.pipeline.modules.values()),
        }

    def shutdown(self) -> dict[str, str]:
        self.pipeline = None
        cleanup_dist_env_and_memory(shutdown_ray=False)
        return {"status": "shutdown_complete"}


class RayMiniMaxH3DisaggregatedRuntime:
    """Own one persistent encoder/decoder actor and one persistent DiT actor."""

    def __init__(
        self,
        fastvideo_args: FastVideoArgs,
        *,
        encoder_node_ip: str,
        dit_node_ip: str,
        ray_address: str | None = None,
    ) -> None:
        assert_ray_available()
        address = ray_address or os.environ.get("RAY_ADDRESS") or "auto"
        if not ray.is_initialized():
            ray.init(address=address, runtime_env=fastvideo_args.ray_runtime_env)
        _validate_topology(encoder_node_ip, dit_node_ip, ray.cluster_resources())

        actor_args = deepcopy(fastvideo_args)
        actor_args.ray_placement_group = None
        actor_args.ray_runtime_env = None
        common_options = {"num_cpus": 0, "num_gpus": 1, "max_restarts": 0}
        self.encoder_node_ip = encoder_node_ip
        self.dit_node_ip = dit_node_ip
        self._closed = False
        self.encoder_decoder = None
        self.dit = None
        try:
            self.encoder_decoder = ray.remote(_MiniMaxH3EncoderDecoderActor).options(
                **common_options,
                resources={
                    _node_resource(encoder_node_ip): 0.001
                },
            ).remote(actor_args)
            self.dit = ray.remote(_MiniMaxH3DiTActor).options(
                **common_options,
                resources={
                    _node_resource(dit_node_ip): 0.001
                },
            ).remote(actor_args)
            self._validate_workers()
        except Exception:
            for actor in (self.encoder_decoder, self.dit):
                if actor is not None:
                    with suppress(Exception):
                        ray.kill(actor, no_restart=True)
            raise

    def _validate_workers(self) -> None:
        health = self.health()
        expected = {
            "encoder_decoder": set(MiniMaxH3EncoderDecoderPipeline._required_config_modules),
            "dit": set(MiniMaxH3DiTPipeline._required_config_modules),
        }
        for receipt in health:
            if not receipt["all_resident"]:
                raise RuntimeError(f"MiniMax-H3 {receipt['role']} worker contains a deferred component.")
            if set(receipt["modules"]) != expected[receipt["role"]]:
                raise RuntimeError(f"MiniMax-H3 {receipt['role']} worker loaded {receipt['modules']}, "
                                   f"expected {tuple(sorted(expected[receipt['role']]))}.")
        actual = {receipt["role"]: receipt["node_ip"] for receipt in health}
        requested = {"encoder_decoder": self.encoder_node_ip, "dit": self.dit_node_ip}
        if actual != requested:
            raise RuntimeError(f"Ray placed MiniMax-H3 roles on {actual}, not the requested nodes {requested}.")

    def health(self) -> list[dict[str, Any]]:
        if self.encoder_decoder is None or self.dit is None:
            raise RuntimeError("MiniMax-H3 disaggregated workers have not been created.")
        return ray.get([self.encoder_decoder.health.remote(), self.dit.health.remote()])

    def submit(self, batch: ForwardBatch, *, request_id: str | None = None):
        """Build a direct actor-to-actor DAG without materializing intermediates on the driver."""
        if self._closed:
            raise RuntimeError("MiniMax-H3 disaggregated runtime is closed.")
        if self.encoder_decoder is None or self.dit is None:
            raise RuntimeError("MiniMax-H3 disaggregated workers have not been created.")
        resolved_request_id = request_id if request_id is not None else _request_id(batch)
        encoded_ref = self.encoder_decoder.encode.remote(batch, resolved_request_id)
        denoised_ref = self.dit.denoise.remote(encoded_ref)
        return self.encoder_decoder.decode.remote(denoised_ref)

    def execute_forward(self, batch: ForwardBatch, *, request_id: str | None = None) -> ForwardBatch:
        return ray.get(self.submit(batch, request_id=request_id))

    async def execute_forward_async(self, batch: ForwardBatch, *, request_id: str | None = None) -> ForwardBatch:
        return await asyncio.to_thread(self.execute_forward, batch, request_id=request_id)

    def iter_forward(self, batches: Iterable[ForwardBatch]) -> Iterator[ForwardBatch]:
        """Run a bounded one-request lookahead pipeline over the two actors.

        Spark A encodes the next request while Spark B denoises the current one.
        Once the current denoise completes, its decode is queued on Spark A and
        the next denoise starts immediately on Spark B. Intermediate ObjectRefs
        are never fetched by the driver.
        """
        iterator = iter(batches)
        try:
            first = next(iterator)
        except StopIteration:
            return

        if self.encoder_decoder is None or self.dit is None:
            raise RuntimeError("MiniMax-H3 disaggregated workers have not been created.")
        encoded_ref = self.encoder_decoder.encode.remote(first, _request_id(first))
        denoised_ref = self.dit.denoise.remote(encoded_ref)
        for next_batch in iterator:
            next_encoded_ref = self.encoder_decoder.encode.remote(next_batch, _request_id(next_batch))
            ray.wait([denoised_ref], num_returns=1, fetch_local=False)
            decoded_ref = self.encoder_decoder.decode.remote(denoised_ref)
            next_denoised_ref = self.dit.denoise.remote(next_encoded_ref)
            yield ray.get(decoded_ref)
            denoised_ref = next_denoised_ref
        yield ray.get(self.encoder_decoder.decode.remote(denoised_ref))

    async def iter_forward_async(self, batches: Iterable[ForwardBatch]) -> AsyncIterator[ForwardBatch]:
        """Asynchronously consume the bounded actor pipeline without blocking the event loop."""
        iterator = iter(self.iter_forward(batches))
        sentinel = object()
        while True:
            result = await asyncio.to_thread(_next_or_sentinel, iterator, sentinel)
            if result is sentinel:
                return
            yield cast(ForwardBatch, result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        actors = [actor for actor in (self.encoder_decoder, self.dit) if actor is not None]
        try:
            ray.get([actor.shutdown.remote() for actor in actors])
        finally:
            for actor in actors:
                with suppress(Exception):
                    ray.kill(actor, no_restart=True)


class MiniMaxH3DisaggregatedExecutor(Executor):
    """VideoGenerator-compatible adapter over the two-role Ray runtime."""

    def _init_executor(self) -> None:
        encoder_ip = self.fastvideo_args.h3_encoder_node_ip
        dit_ip = self.fastvideo_args.h3_dit_node_ip
        if encoder_ip is None or dit_ip is None:
            raise ValueError("Set h3_encoder_node_ip and h3_dit_node_ip for component disaggregation.")
        self.runtime = RayMiniMaxH3DisaggregatedRuntime(
            self.fastvideo_args,
            encoder_node_ip=encoder_ip,
            dit_node_ip=dit_ip,
            ray_address=self.fastvideo_args.h3_ray_address,
        )

    def execute_forward(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        return self.runtime.execute_forward(forward_batch)

    async def execute_forward_async(self, forward_batch: ForwardBatch) -> ForwardBatch:
        return await self.runtime.execute_forward_async(forward_batch)

    def iter_forward(self, batches: Iterable[ForwardBatch]) -> Iterator[ForwardBatch]:
        return self.runtime.iter_forward(batches)

    def iter_forward_async(self, batches: Iterable[ForwardBatch]) -> AsyncIterator[ForwardBatch]:
        return self.runtime.iter_forward_async(batches)

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None,
                         strength: float = 1.0,
                         accumulate: bool = False) -> None:
        receipt = ray.get(self.runtime.dit.set_lora_adapter.remote(lora_nickname, lora_path, strength, accumulate))
        if receipt.get("status") != "lora_adapter_set":
            raise RuntimeError(f"MiniMax-H3 DiT worker rejected the LoRA adapter: {receipt}.")

    def unmerge_lora_weights(self) -> None:
        ray.get(self.runtime.dit.unmerge_lora_weights.remote())

    def merge_lora_weights(self) -> None:
        ray.get(self.runtime.dit.merge_lora_weights.remote())

    def collective_rpc(self,
                       method,
                       timeout: float | None = None,
                       args: tuple = (),
                       kwargs: dict[str, Any] | None = None) -> list[Any]:
        raise NotImplementedError("Component-disaggregated H3 has role-specific RPC; use runtime actor handles.")

    def set_log_queue(self, log_queue: Queue | None) -> None:
        # multiprocessing.Queue cannot cross Ray nodes. Actor logs remain in the
        # Ray session log, matching RayDistributedExecutor's behavior.
        self._log_queue = log_queue

    def clear_log_queue(self) -> None:
        self._log_queue = None

    def shutdown(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.shutdown()


__all__ = [
    "MiniMaxH3DisaggregatedExecutor",
    "RayMiniMaxH3DisaggregatedRuntime",
]
