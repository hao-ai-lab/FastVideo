# pyright: reportArgumentType=false, reportMissingImports=false
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from gpu_pool import GPUPool, get_available_gpus
from session_logger import SessionEventLogger

from config import (
    DEVTOOLS_ENABLED,
    FRONTEND_STATIC_DIR_CANDIDATES,
    PROMPT_SAFETY_ENABLED,
    SESSION_LOG_ROOT,
)
from prompt_enhancer import PromptEnhancer
from prompt_safety import PromptSafetyFilter

import runtime
from routes.health import (
    router as health_router, )
from routes.presets import (
    prompt_config_router,
    curated_presets_router,
)
from session.controller import SessionController
if DEVTOOLS_ENABLED:
    pass


class _HeartbeatAccessLogFilter(logging.Filter):
    """Drop noisy access logs for frequent health/readiness probes."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return ('"GET /healthz ' not in message and '"GET /readyz ' not in message)


def _install_heartbeat_log_filter() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    for existing in access_logger.filters:
        if isinstance(existing, _HeartbeatAccessLogFilter):
            return
    access_logger.addFilter(_HeartbeatAccessLogFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("Starting server...")

    # Get available GPUs
    gpu_ids = get_available_gpus()
    print(f"Selected GPU ids: {gpu_ids}")

    # Initialize GPU pool (spawns subprocess per GPU)
    runtime.gpu_pool = GPUPool(gpu_ids)
    await runtime.gpu_pool.initialize()

    runtime.prompt_enhancer = PromptEnhancer()
    runtime.session_event_logger = SessionEventLogger(Path(SESSION_LOG_ROOT))
    runtime.prompt_safety_filter = (PromptSafetyFilter() if PROMPT_SAFETY_ENABLED else None)
    if runtime.prompt_safety_filter is not None:
        print("Prompt safety filter enabled")

    print("Server started")
    yield

    print("Shutting down server...")
    await runtime.gpu_pool.shutdown()
    runtime.prompt_safety_filter = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(prompt_config_router)
if DEVTOOLS_ENABLED:
    app.include_router(curated_presets_router)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    controller = SessionController(
        ws=websocket,
        gpu_pool=runtime.gpu_pool,
        prompt_enhancer=runtime.prompt_enhancer,
        prompt_safety_filter=runtime.prompt_safety_filter,
        session_event_logger=runtime.session_event_logger,
    )
    await controller.run()


# Serve server-side assets (images, etc.)
server_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/server-assets", StaticFiles(directory=server_dir), name="server-assets")

# Serve an exported frontend bundle when present.
for static_dir in FRONTEND_STATIC_DIR_CANDIDATES:
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
        break


def cli() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()

    _install_heartbeat_log_filter()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    cli()
