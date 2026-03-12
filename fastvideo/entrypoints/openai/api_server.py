# Adapted from SGLang
# (https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py)

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastvideo.entrypoints.openai.state import (
    DEFAULT_OUTPUT_DIR,
    clear_state,
    set_state,
)
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model on startup, clean up on shutdown"""
    args: FastVideoArgs = app.state.fastvideo_args
    output_dir: str = app.state.output_dir

    logger.info("Loading model from %s ...", args.model_path)
    generator = VideoGenerator.from_fastvideo_args(args)
    logger.info("Model loaded successfully.")

    set_state(generator, args, output_dir)

    yield  # server is running

    logger.info("Shutting down — releasing model resources ...")
    generator.shutdown()
    clear_state()
    logger.info("Shutdown complete.")


def create_app(
    fastvideo_args: FastVideoArgs,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> FastAPI:
    """Build the FastAPI application with all routers mounted"""

    app = FastAPI(
        title="FastVideo OpenAI-Compatible API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.fastvideo_args = fastvideo_args
    app.state.output_dir = output_dir

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


def _parse_args() -> tuple[FastVideoArgs, str, int, str]:
    """Parse CLI arguments and return (FastVideoArgs, host, port, output_dir)"""
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="FastVideo OpenAI-compatible API server")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser = FastVideoArgs.add_cli_args(parser)

    args = parser.parse_args()
    host = args.host
    port = args.port
    output_dir = args.output_dir

    # Build FastVideoArgs from the remaining CLI args
    excluded = {"host", "port", "output_dir", "subparser", "config", "dispatch_function"}
    cli_kwargs = {k: v for k, v in vars(args).items() if k not in excluded and v is not None}
    fastvideo_args = FastVideoArgs.from_kwargs(**cli_kwargs)
    return fastvideo_args, host, port, output_dir


def run_server(
    fastvideo_args: FastVideoArgs,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    """Create the app and run it with uvicorn"""
    app = create_app(fastvideo_args, output_dir=output_dir)

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
    fastvideo_args, host, port, output_dir = _parse_args()
    run_server(fastvideo_args, host, port, output_dir=output_dir)
