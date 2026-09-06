# SPDX-License-Identifier: Apache-2.0
"""Ray data-plane for component-disaggregated MiniMax H3 inference."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass
import os
from queue import Queue
import time
from typing import Any, cast
from uuid import uuid4

import torch

import fastvideo.envs as envs
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
from fastvideo.profiler import nvtx_range
from fastvideo.utils import get_ip, get_open_port
from fastvideo.worker.executor import Executor
from fastvideo.worker.ray_utils import assert_ray_available, ray

logger = init_logger(__name__)


def _h3_state_tensors(state: Any) -> Iterator[tuple[str, torch.Tensor]]:
    for field_name in ("prompt_embeds", "video_latents", "audio_latents"):
        tensor = getattr(state, field_name, None)
        if tensor is not None:
            yield field_name, tensor

    layout = getattr(state, "layout", None)
    if layout is not None:
        for field_name in ("position_ids", "token_tags", "text_indices", "video_indices", "audio_indices"):
            tensor = getattr(layout, field_name, None)
            if tensor is not None:
                yield f"layout.{field_name}", tensor


def _log_h3_state(name: str, state: Any) -> int:
    total_bytes = 0
    for field_name, tensor in _h3_state_tensors(state):
        size = tensor.numel() * tensor.element_size()
        total_bytes += size
        logger.info(
            "[RAY_PAYLOAD] %s.%s: %.2f MB shape=%s dtype=%s device=%s request_id=%s",
            name,
            field_name,
            size / 1e6,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            state.request_id,
        )

    logger.info(
        "[RAY_PAYLOAD] %s TOTAL: %.2f MB request_id=%s (logical tensor bytes)",
        name,
        total_bytes / 1e6,
        state.request_id,
    )
    return total_bytes


@dataclass(frozen=True)
class _H3TransferRef:
    """Keep the ref nested so Ray does not fetch it before actor entry."""

    ref: Any
    request_id: str


def _synchronize_h3_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _receive_h3_state(state: Any, direction: str) -> Any:
    if not isinstance(state, _H3TransferRef):
        return state

    # A receiver-local clock avoids clock skew between the two Sparks. Drain
    # earlier device work before timing, then wait for production WITHOUT
    # fetching locally so producer compute is not counted as object fetch.
    _synchronize_h3_device()
    started = time.perf_counter()
    with nvtx_range(f"h3.{direction}.source_wait"):
        ray.wait([state.ref], num_returns=1, fetch_local=False)
    source_ready = time.perf_counter()
    with nvtx_range(f"h3.{direction}.object_fetch"):
        ray.wait([state.ref], num_returns=1, fetch_local=True)
    fetched = time.perf_counter()
    with nvtx_range(f"h3.{direction}.materialize"):
        result = ray.get(state.ref)
        _synchronize_h3_device()
    materialized = time.perf_counter()

    tensor_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in _h3_state_tensors(result))
    object_fetch_s = fetched - source_ready
    # This is an effective logical-payload rate, not measured NIC bandwidth.
    rate = f"{tensor_bytes / 1e6 / object_fetch_s:.2f}" if object_fetch_s > 0 else "n/a"
    logger.info(
        "[RAY_TRANSFER] request_id=%s direction=%s tensor_bytes=%d "
        "source_wait_s=%.6f object_fetch_s=%.6f materialize_s=%.6f "
        "receive_s=%.6f object_fetch_mb_s=%s",
        state.request_id,
        direction,
        tensor_bytes,
        source_ready - started,
        object_fetch_s,
        materialized - fetched,
        materialized - source_ready,
        rate,
    )
    return result


@contextmanager
def _h3_stage_timer(stage: str, request_id: str, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    _synchronize_h3_device()
    started = time.perf_counter()
    with nvtx_range(f"h3.{stage}"):
        yield
        _synchronize_h3_device()
    logger.info("[RAY_STAGE] request_id=%s stage=%s elapsed_s=%.6f", request_id, stage, time.perf_counter() - started)


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

    def __init__(self, fastvideo_args: FastVideoArgs, profile_transfers: bool = False) -> None:
        self._profile_transfers = profile_transfers
        _bind_single_gpu_process()
        args = _resident_role_args(fastvideo_args, role="encoder_decoder")
        pipeline_cls = MiniMaxH3RefEncoderDecoderPipeline if _is_ref2va(args) else MiniMaxH3EncoderDecoderPipeline
        self.pipeline = pipeline_cls(args.model_path, args)
        self.pipeline.post_init()

    def encode(self, batch: ForwardBatch, request_id: str) -> MiniMaxH3EncodedState:
        with _h3_stage_timer("encode", request_id, self._profile_transfers):
            return self.pipeline.encode(batch, request_id=request_id)

    def decode(self, state: MiniMaxH3DenoisedState | _H3TransferRef) -> ForwardBatch:
        state = _receive_h3_state(state, "B_TO_A")
        with _h3_stage_timer("decode", state.request_id, self._profile_transfers):
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

    def __init__(self, fastvideo_args: FastVideoArgs, profile_transfers: bool = False) -> None:
        self._profile_transfers = profile_transfers
        _bind_single_gpu_process()
        args = _resident_role_args(fastvideo_args, role="dit")
        pipeline_cls = MiniMaxH3RefDiTPipeline if _is_ref2va(args) else MiniMaxH3DiTPipeline
        self.pipeline = pipeline_cls(args.model_path, args)
        self.pipeline.post_init()

    def health(self) -> dict[str, Any]:
        modules = tuple(sorted(self.pipeline.modules))
        return {
            "ready": True,
            "role": "dit",
            "node_ip": get_ip(),
            "modules": modules,
            "all_resident": all(not is_lazy_module(module) for module in self.pipeline.modules.values()),
        }

    def denoise(self, state: MiniMaxH3EncodedState | _H3TransferRef) -> MiniMaxH3DenoisedState:
        state = _receive_h3_state(state, "A_TO_B")
        _log_h3_state("A_TO_B encoded", state)
        with _h3_stage_timer("denoise", state.request_id, self._profile_transfers):
            result = self.pipeline.denoise(state)
        _log_h3_state("B_TO_A denoised", result)
        return result


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

        self._profile_transfers = envs.FASTVIDEO_H3_PROFILE_TRANSFERS
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
            ).remote(actor_args, self._profile_transfers)
            self.dit = ray.remote(_MiniMaxH3DiTActor).options(
                **common_options,
                resources={
                    _node_resource(dit_node_ip): 0.001
                },
            ).remote(actor_args, self._profile_transfers)
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

    def _transfer_arg(self, ref: Any, request_id: str) -> Any:
        return _H3TransferRef(ref, request_id) if self._profile_transfers else ref

    def submit(self, batch: ForwardBatch, *, request_id: str | None = None):
        """Build a direct actor-to-actor DAG without materializing intermediates on the driver."""
        if self._closed:
            raise RuntimeError("MiniMax-H3 disaggregated runtime is closed.")
        if self.encoder_decoder is None or self.dit is None:
            raise RuntimeError("MiniMax-H3 disaggregated workers have not been created.")
        resolved_request_id = request_id if request_id is not None else _request_id(batch)
        encoded_ref = self.encoder_decoder.encode.remote(batch, resolved_request_id)
        denoised_ref = self.dit.denoise.remote(self._transfer_arg(encoded_ref, resolved_request_id))
        return self.encoder_decoder.decode.remote(self._transfer_arg(denoised_ref, resolved_request_id))

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
        request_id = _request_id(first)
        encoded_ref = self.encoder_decoder.encode.remote(first, request_id)
        denoised_ref = self.dit.denoise.remote(self._transfer_arg(encoded_ref, request_id))
        for next_batch in iterator:
            next_request_id = _request_id(next_batch)
            next_encoded_ref = self.encoder_decoder.encode.remote(next_batch, next_request_id)
            ray.wait([denoised_ref], num_returns=1, fetch_local=False)
            decoded_ref = self.encoder_decoder.decode.remote(self._transfer_arg(denoised_ref, request_id))
            next_denoised_ref = self.dit.denoise.remote(self._transfer_arg(next_encoded_ref, next_request_id))
            yield ray.get(decoded_ref)
            denoised_ref = next_denoised_ref
            request_id = next_request_id
        yield ray.get(self.encoder_decoder.decode.remote(self._transfer_arg(denoised_ref, request_id)))

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
