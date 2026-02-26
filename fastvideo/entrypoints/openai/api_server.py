# Adapted from SGLang (https://github.com/sgl-project/sglang)

import argparse
import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger

logger = init_logger(__name__)
_generator: VideoGenerator | None = None
_fastvideo_args: FastVideoArgs | None = None


def get_generator() -> VideoGenerator:
    """Return the global VideoGenerator instance (set during startup)"""
    assert _generator is not None, "Server not initialized — generator is None"
    return _generator


def get_server_args() -> FastVideoArgs:
    """Return the global FastVideoArgs (set during startup)"""
    assert _fastvideo_args is not None, "Server not initialized — args is None"
    return _fastvideo_args


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model on startup, clean up on shutdown"""
    global _generator, _fastvideo_args

    args: FastVideoArgs = app.state.fastvideo_args
    _fastvideo_args = args

    logger.info("Loading model from %s ...", args.model_path)
    _generator = VideoGenerator.from_fastvideo_args(args)
    logger.info("Model loaded successfully.")

    yield  # server is running

    logger.info("Shutting down — releasing model resources ...")
    if _generator is not None:
        _generator.shutdown()
        _generator = None
    logger.info("Shutdown complete.")


def create_app(fastvideo_args: FastVideoArgs) -> FastAPI:
    """Build the FastAPI application with all routers mounted"""

    app = FastAPI(
        title="FastVideo OpenAI-Compatible API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.fastvideo_args = fastvideo_args

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and mount routers
    from fastvideo.entrypoints.openai.common_api import router as common_router
    from fastvideo.entrypoints.openai.image_api import router as image_router
    from fastvideo.entrypoints.openai.video_api import router as video_router

    app.include_router(common_router)
    app.include_router(video_router)
    app.include_router(image_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def _parse_args() -> tuple[FastVideoArgs, str, int]:
    """Parse CLI arguments and return (FastVideoArgs, host, port)"""
    from fastvideo.configs.sample import SamplingParam
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="FastVideo OpenAI-compatible API server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser = FastVideoArgs.add_cli_args(parser)

    args = parser.parse_args()
    host = args.host
    port = args.port

    # Build FastVideoArgs from the remaining CLI args
    cli_kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in ("host", "port", "subparser", "config", "dispatch_function")
        and v is not None
    }
    fastvideo_args = FastVideoArgs.from_kwargs(**cli_kwargs)
    return fastvideo_args, host, port


def run_server(fastvideo_args: FastVideoArgs, host: str = "0.0.0.0", port: int = 8000):
    """Create the app and run it with uvicorn"""
    app = create_app(fastvideo_args)

    logger.info("Starting FastVideo server on %s:%d", host, port)
    logger.info("Model: %s", fastvideo_args.model_path)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    fastvideo_args, host, port = _parse_args()
    run_server(fastvideo_args, host, port)
