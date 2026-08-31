# Adapted from SGLang
# (https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py)

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastvideo.api.presets import validate_preset_selection
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.openai.state import (
    DEFAULT_OUTPUT_DIR,
    clear_state,
    set_state,
)
from fastvideo.entrypoints.openai.serving_engine import OpenAIServingEngine
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.registry import get_preset_selection

logger = init_logger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def _validate_default_request_against_preset(
    default_request: GenerationRequest,
    model_path: str,
) -> None:
    """Validate ``default_request.stage_overrides`` against the model's preset.

    Called once at server startup from :func:`run_server`. The
    ``default_request`` is static server config, so validation results are
    invariant across requests — there's no reason to re-run per request.
    """
    if not default_request.stage_overrides:
        return
    preset_name, model_family = get_preset_selection(model_path)
    if preset_name is None or model_family is None:
        return
    validate_preset_selection(
        preset_name,
        model_family,
        stage_overrides=default_request.stage_overrides,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model on startup, clean up on shutdown"""
    args: FastVideoArgs = app.state.fastvideo_args
    output_dir: str = app.state.output_dir
    served_model_name: str | None = app.state.served_model_name
    default_request: GenerationRequest | None = getattr(app.state, "default_request", None)

    logger.info("Loading model from %s ...", args.model_path)
    generator = VideoGenerator.from_fastvideo_args(args)
    serving_engine = OpenAIServingEngine(generator)
    logger.info("Model loaded successfully.")

    set_state(
        generator,
        serving_engine,
        args,
        output_dir,
        default_request=default_request,
        served_model_name=served_model_name,
    )

    try:
        yield  # server is running
    finally:
        logger.info("Shutting down — releasing model resources ...")
        from fastvideo.entrypoints.openai.video_api import shutdown_video_jobs

        await shutdown_video_jobs()
        await serving_engine.shutdown()
        clear_state()
        logger.info("Shutdown complete.")


def create_app(
    fastvideo_args: FastVideoArgs,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    default_request: GenerationRequest | None = None,
    served_model_name: str | None = None,
) -> FastAPI:
    """Build the FastAPI application with all routers mounted"""

    app = FastAPI(
        title="FastVideo OpenAI-Compatible API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.fastvideo_args = fastvideo_args
    app.state.output_dir = output_dir
    app.state.default_request = default_request
    app.state.served_model_name = served_model_name

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def openai_http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        """Return the error envelope consumed by OpenAI-compatible clients."""
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            headers=exc.headers,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                    "param": None,
                    "code": exc.status_code,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def openai_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "param": None,
                "code": 400,
            }},
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
        from fastvideo.entrypoints.openai.state import get_serving_engine

        engine = get_serving_engine()
        if not engine.healthy:
            raise HTTPException(status_code=503, detail=engine.unhealthy_reason or "generation engine is unhealthy")
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
    default_request: GenerationRequest | None = None,
    served_model_name: str | None = None,
):
    """Create the app and run it with uvicorn"""
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")
    if default_request is not None:
        _validate_default_request_against_preset(default_request, fastvideo_args.model_path)

    app = create_app(
        fastvideo_args,
        output_dir=output_dir,
        default_request=default_request,
        served_model_name=served_model_name,
    )

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
