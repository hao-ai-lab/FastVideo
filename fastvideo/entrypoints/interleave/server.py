# SPDX-License-Identifier: Apache-2.0
"""InterleaveThinker-compatible FastVideo image generation service."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.interleave.generator import (
    FastVideoImageGeneratorBackend,
    ImageGeneratorBackend,
)
from fastvideo.entrypoints.interleave.schema import (
    InterleaveEditRequest,
    InterleaveEditResponse,
)
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8011
DEFAULT_OUTPUT_DIR = "outputs/interleave"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    generator = None
    if app.state.backend is None:
        args: FastVideoArgs = app.state.fastvideo_args
        logger.info("Loading Interleave FastVideo backend from %s ...", args.model_path)
        generator = VideoGenerator.from_fastvideo_args(args)
        app.state.backend = FastVideoImageGeneratorBackend(
            generator,
            output_dir=app.state.output_dir,
            default_request=app.state.default_request,
        )
        logger.info("Interleave FastVideo backend loaded.")

    yield

    if generator is not None:
        logger.info("Shutting down Interleave FastVideo backend ...")
        generator.shutdown()


def build_app(
    *,
    fastvideo_args: FastVideoArgs | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    default_request: GenerationRequest | None = None,
    backend: ImageGeneratorBackend | None = None,
) -> FastAPI:
    if backend is None and fastvideo_args is None:
        raise ValueError("build_app requires either `backend` or `fastvideo_args`")

    app = FastAPI(
        title="FastVideo Interleave Compatibility API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.fastvideo_args = fastvideo_args
    app.state.output_dir = output_dir
    app.state.default_request = default_request
    app.state.backend = backend

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/edit", response_model=InterleaveEditResponse)
    async def edit(request: InterleaveEditRequest) -> InterleaveEditResponse:
        return await _run_edit(app, request)

    @app.post("/generate", response_model=InterleaveEditResponse)
    async def generate(request: InterleaveEditRequest) -> InterleaveEditResponse:
        return await _run_edit(app, request)

    @app.post("/v1/interleave/edit", response_model=InterleaveEditResponse)
    async def interleave_edit(request: InterleaveEditRequest) -> InterleaveEditResponse:
        return await _run_edit(app, request)

    return app


async def _run_edit(
    app: FastAPI,
    request: InterleaveEditRequest,
) -> InterleaveEditResponse:
    request_id = uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    backend: ImageGeneratorBackend = app.state.backend
    try:
        generated = await loop.run_in_executor(None, lambda: backend.generate(request, request_id=request_id))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Interleave edit request %s failed", request_id)
        return InterleaveEditResponse(
            success=False,
            prompt=request.prompt,
            error=str(exc),
            metadata={"request_id": request_id},
        )

    metadata = dict(generated.metadata)
    metadata.setdefault("request_id", request_id)
    return InterleaveEditResponse(
        success=True,
        edited_image=generated.image_base64,
        file_path=generated.file_path,
        prompt=generated.prompt,
        inference_time_s=generated.inference_time_s,
        metadata=metadata,
    )


def run_server(
    fastvideo_args: FastVideoArgs,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    default_request: GenerationRequest | None = None,
) -> None:
    app = build_app(
        fastvideo_args=fastvideo_args,
        output_dir=output_dir,
        default_request=default_request,
    )
    logger.info("Starting FastVideo Interleave server on %s:%d", host, port)
    logger.info("Model: %s", fastvideo_args.model_path)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=300,
    )
