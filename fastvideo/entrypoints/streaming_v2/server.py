# SPDX-License-Identifier: Apache-2.0
"""The streaming_v2 FastAPI app: it owns one generator and runs one generation at a time."""
from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import FastAPI

from fastvideo.api.schema import GenerationRequest, ServeConfig
from fastvideo.entrypoints.streaming_v2.routes import capabilities, generate, health


class GeneratorLike(Protocol):
    """The part of :class:`fastvideo.VideoGenerator` that the server calls."""

    def generate(self, request: GenerationRequest) -> Any:
        ...


@dataclass
class ServerState:
    """The server's generator, its operator defaults, and its generation lock."""

    generator: GeneratorLike
    # Operator baseline from the serve config; only fields the operator wrote apply.
    default_request: GenerationRequest | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def generate(
        self,
        request: GenerationRequest,
        reference_dir: str | None = None,
    ) -> tuple[Any, float]:
        """Run one generation and return its result and duration in milliseconds.

        The call cannot be interrupted, so the lock stays held until it returns even when the caller is
        cancelled; otherwise a second generation could start on the same pipeline. The generation task, not the
        caller, removes ``reference_dir``, the directory of the reference files that ``request`` reads.
        """

        async def run_locked() -> tuple[Any, float]:
            try:
                async with self.lock:
                    started = time.perf_counter()
                    result = await asyncio.to_thread(self.generator.generate, request)
                    return result, (time.perf_counter() - started) * 1000.0
            finally:
                if reference_dir is not None:
                    shutil.rmtree(reference_dir, ignore_errors=True)

        task = asyncio.ensure_future(run_locked())
        task.add_done_callback(_retrieve_exception)
        return await asyncio.shield(task)


def _retrieve_exception(task: asyncio.Future) -> None:
    """Mark a shielded task's exception as seen when its caller was cancelled and never awaited it."""
    if not task.cancelled():
        task.exception()


def create_app(
    generator: GeneratorLike,
    default_request: GenerationRequest | None = None,
) -> FastAPI:
    """Build the app around an already created generator."""
    app = FastAPI()
    app.state.streaming_v2 = ServerState(generator=generator, default_request=default_request)
    app.include_router(generate.router)
    app.include_router(capabilities.router)
    app.include_router(health.router)
    return app


def run_server(serve_config: ServeConfig) -> None:
    """Create the generator from the serve config, serve until stopped, then shut the generator down."""
    import uvicorn

    from fastvideo import VideoGenerator  # lazy: loading a model is slow

    generator = VideoGenerator.from_pretrained(config=serve_config.generator)
    try:
        uvicorn.run(
            create_app(generator, serve_config.default_request),
            host=serve_config.server.host,
            port=serve_config.server.port,
        )
    finally:
        generator.shutdown()
