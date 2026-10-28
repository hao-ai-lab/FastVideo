# SPDX-License-Identifier: Apache-2.0
"""The streaming_v2 FastAPI app: it owns one generator and runs one generation at a time."""
from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import FastAPI

from fastvideo.api.schema import GenerationRequest, ServeConfig, StreamingV2LoRAConfig
from fastvideo.entrypoints.streaming_v2.lora import LoraRequestError, LoraRuntime, adapters_attach_as_side_branch
from fastvideo.entrypoints.streaming_v2.messages import LoraSelection
from fastvideo.entrypoints.streaming_v2.routes import capabilities, generate, health


class GeneratorLike(Protocol):
    """The part of :class:`fastvideo.VideoGenerator` that the server calls."""

    def generate(self, request: GenerationRequest) -> Any:
        ...

    def set_lora_adapter(self, lora_nickname: str, lora_path: str | None = None, strength: float = 1.0) -> None:
        ...


@dataclass
class ServerState:
    """The server's generator, its operator defaults, its LoRA allow-list, and its generation lock."""

    generator: GeneratorLike
    # Operator baseline from the serve config; only fields the operator wrote apply.
    default_request: GenerationRequest | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # ``None`` when the serve config declares no adapter; every ``lora`` selection is then refused.
    lora: LoraRuntime | None = None

    async def generate(
        self,
        request: GenerationRequest,
        lora: LoraSelection | None = None,
        reference_dir: str | None = None,
    ) -> tuple[Any, float]:
        """Apply the LoRA selection, run one generation, and return its result and duration in milliseconds.

        The call cannot be interrupted, so the lock stays held until it returns even when the caller is
        cancelled; otherwise a second generation could start on the same pipeline. The generation task, not the
        caller, removes ``reference_dir``, the directory of the reference files that ``request`` reads.
        """

        async def run_locked() -> tuple[Any, float]:
            try:
                async with self.lock:
                    self.validate_lora(lora)
                    if self.lora is not None:
                        await asyncio.to_thread(self.lora.apply, self.generator, lora)
                    started = time.perf_counter()
                    result = await asyncio.to_thread(self.generator.generate, request)
                    return result, (time.perf_counter() - started) * 1000.0
            finally:
                if reference_dir is not None:
                    shutil.rmtree(reference_dir, ignore_errors=True)

        task = asyncio.ensure_future(run_locked())
        task.add_done_callback(_retrieve_exception)
        return await asyncio.shield(task)

    def validate_lora(self, lora: LoraSelection | None) -> None:
        """Refuse an adapter selection this server cannot serve, before a stream starts."""
        if lora is None:
            return
        if self.lora is None:
            raise LoraRequestError("This server declares no LoRA adapter; omit the lora field.")
        self.lora.validate(lora)


def _retrieve_exception(task: asyncio.Future) -> None:
    """Mark a shielded task's exception as seen when its caller was cancelled and never awaited it."""
    if not task.cancelled():
        task.exception()


def create_app(
    generator: GeneratorLike,
    default_request: GenerationRequest | None = None,
    lora_config: StreamingV2LoRAConfig | None = None,
    lora_runtime_switching: bool = False,
) -> FastAPI:
    """Build the app around an already created generator."""
    app = FastAPI()
    lora = None
    if lora_config is not None and lora_config.adapters:
        lora = LoraRuntime(config=lora_config, runtime_switching=lora_runtime_switching)
    app.state.streaming_v2 = ServerState(generator=generator, default_request=default_request, lora=lora)
    app.include_router(generate.router)
    app.include_router(capabilities.router)
    app.include_router(health.router)
    return app


def run_server(serve_config: ServeConfig) -> None:
    """Create the generator from the serve config, serve until stopped, then shut the generator down."""
    import uvicorn

    from fastvideo import VideoGenerator  # lazy: loading a model is slow

    streaming_v2 = serve_config.streaming_v2
    lora_config = None if streaming_v2 is None else streaming_v2.lora
    generator = VideoGenerator.from_pretrained(config=serve_config.generator)
    try:
        uvicorn.run(
            create_app(generator, serve_config.default_request, lora_config,
                       adapters_attach_as_side_branch(serve_config.generator)),
            host=serve_config.server.host,
            port=serve_config.server.port,
        )
    finally:
        generator.shutdown()
