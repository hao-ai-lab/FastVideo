# SPDX-License-Identifier: Apache-2.0
"""Shared asynchronous execution substrate for OpenAI-compatible routes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.openai.lora import LoraRuntime, LoraSelection
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest

_T = TypeVar("_T")


class ServingGenerator(Protocol):

    def generate(self, request: GenerationRequest) -> Any:
        ...

    def shutdown(self) -> None:
        ...


class OpenAIServingEngine:
    """Own generator lifecycle and serialize access to its mutable pipeline.

    FastVideo pipelines contain request-mutated sampling state and some LoRA
    implementations merge weights in place. Running two Python threads through
    one pipeline is therefore unsafe even if the HTTP layer accepts requests
    concurrently. This engine gives every OpenAI route one model-agnostic async
    entrypoint while preserving that invariant. A future scheduler can replace
    the lock without changing the transport contract.
    """

    def __init__(self,
                 generator: ServingGenerator,
                 video_request_validator: Callable[[VideoGenerationRequest], None] | None = None,
                 lora_runtime: LoraRuntime | None = None) -> None:
        self._generator = generator
        self._lora_runtime = lora_runtime
        self._video_request_validator = video_request_validator
        self._generation_lock = asyncio.Lock()
        self._closed = False
        self._unhealthy_reason: str | None = None

    @property
    def generator(self) -> ServingGenerator:
        return self._generator

    @property
    def lora_runtime(self) -> LoraRuntime | None:
        return self._lora_runtime

    def validate_video_request(self, request: VideoGenerationRequest) -> None:
        if self._video_request_validator is not None:
            self._video_request_validator(request)

    @property
    def closed(self) -> bool:
        return self._closed

    def health_status(self) -> tuple[bool, str | None]:
        """Probe executor liveness once and return ``(healthy, reason)``.

        The executor probe may make a synchronous control-plane call, so callers
        on the event loop must run it in a worker thread.
        """
        if self._closed or self._unhealthy_reason is not None:
            return False, self._unhealthy_reason or "one or more generation workers are not alive"
        executor = getattr(self._generator, "executor", None)
        if executor is None or executor.healthy:
            return True, None
        return False, "one or more generation workers are not alive"

    @property
    def healthy(self) -> bool:
        return self.health_status()[0]

    @property
    def unhealthy_reason(self) -> str | None:
        return self.health_status()[1]

    async def generate(
        self,
        request: GenerationRequest,
        *,
        on_start: Callable[[], Awaitable[None]] | None = None,
        lora: LoraSelection | None = None,
    ) -> Any:
        """Generate one typed request without blocking the event loop."""
        return await self.run_serialized(self._generator.generate, request, on_start=on_start, lora=lora)

    async def run_serialized(
        self,
        function: Callable[..., _T],
        *args: Any,
        on_start: Callable[[], Awaitable[None]] | None = None,
        lora: LoraSelection | None = None,
        **kwargs: Any,
    ) -> _T:
        """Run a synchronous pipeline operation under the serving lock.

        With a LoRA allow-list, the adapter change runs in the same locked call as ``function``; ``lora=None``
        renders the base model.
        """
        if self._closed:
            raise RuntimeError("FastVideo serving engine is shutting down")
        async with self._generation_lock:
            if self._closed:
                raise RuntimeError("FastVideo serving engine is shutting down")
            if on_start is not None:
                await on_start()
            call = function if self._lora_runtime is None else self._with_lora(function, lora)
            worker = asyncio.create_task(asyncio.to_thread(call, *args, **kwargs))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Python cannot stop a running worker thread. Keep the lock
                # until the pipeline call really exits so cancellation cannot
                # expose mutable model state to a second request.
                await self._wait_after_cancellation(worker)
                raise
            except (BrokenPipeError, EOFError) as error:
                self._unhealthy_reason = str(error)
                raise

    def _with_lora(self, function: Callable[..., _T], selection: LoraSelection | None) -> Callable[..., _T]:
        runtime = self._lora_runtime
        assert runtime is not None

        def call(*args: Any, **kwargs: Any) -> _T:
            runtime.apply(self._generator, selection)
            try:
                return function(*args, **kwargs)
            except (BrokenPipeError, EOFError):
                raise
            except Exception:
                runtime.invalidate()
                raise

        return call

    async def run_async_serialized(self, function: Callable[[], Awaitable[_T]]) -> _T:
        """Run an async operation under the same pipeline lock."""
        if self._closed:
            raise RuntimeError("FastVideo serving engine is shutting down")
        async with self._generation_lock:
            if self._closed:
                raise RuntimeError("FastVideo serving engine is shutting down")
            worker: asyncio.Future[_T] = asyncio.ensure_future(function())
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                await self._wait_after_cancellation(worker)
                raise

    @staticmethod
    async def _wait_after_cancellation(worker: asyncio.Future[Any]) -> None:
        """Keep waiting for an uninterruptible worker despite repeated cancellation."""
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()

    async def shutdown(self) -> None:
        """Stop accepting requests and release the generator after in-flight work."""
        self._closed = True
        async with self._generation_lock:
            await asyncio.to_thread(self._generator.shutdown)


__all__ = ["OpenAIServingEngine"]
