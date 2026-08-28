# SPDX-License-Identifier: Apache-2.0
"""Shared asynchronous execution substrate for OpenAI-compatible routes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.video_generator import VideoGenerator

_T = TypeVar("_T")


class OpenAIServingEngine:
    """Own generator lifecycle and serialize access to its mutable pipeline.

    FastVideo pipelines contain request-mutated sampling state and some LoRA
    implementations merge weights in place. Running two Python threads through
    one pipeline is therefore unsafe even if the HTTP layer accepts requests
    concurrently. This engine gives every OpenAI route one model-agnostic async
    entrypoint while preserving that invariant. A future scheduler can replace
    the lock without changing the transport contract.
    """

    def __init__(self, generator: VideoGenerator) -> None:
        self._generator = generator
        self._generation_lock = asyncio.Lock()
        self._closed = False

    @property
    def generator(self) -> VideoGenerator:
        return self._generator

    @property
    def closed(self) -> bool:
        return self._closed

    async def generate(self, request: GenerationRequest) -> Any:
        """Generate one typed request without blocking the event loop."""
        return await self.run_serialized(self._generator.generate, request)

    async def run_serialized(self, function: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """Run a synchronous pipeline operation under the serving lock."""
        if self._closed:
            raise RuntimeError("FastVideo serving engine is shutting down")
        async with self._generation_lock:
            if self._closed:
                raise RuntimeError("FastVideo serving engine is shutting down")
            worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Python cannot stop a running worker thread. Keep the lock
                # until the pipeline call really exits so cancellation cannot
                # expose mutable model state to a second request.
                await asyncio.shield(worker)
                raise

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
                await asyncio.shield(worker)
                raise

    async def shutdown(self) -> None:
        """Stop accepting requests and release the generator after in-flight work."""
        self._closed = True
        async with self._generation_lock:
            await asyncio.to_thread(self._generator.shutdown)


__all__ = ["OpenAIServingEngine"]
