# pyright: reportArgumentType=false, reportMissingImports=false
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastvideo.entrypoints.streaming import build_health_router
from dreamverse.gpu_pool import GPUPool, get_available_gpus
from dreamverse.session_logger import SessionEventLogger

from dreamverse.config import (
    AVAILABLE_LORAS,
    DEVTOOLS_ENABLED,
    FRONTEND_STATIC_DIR_CANDIDATES,
    PROMPT_SAFETY_ENABLED,
    SESSION_LOG_ROOT,
    _available_styles_for_active_model,
    _resolve_lora_spec,
)
from dreamverse.prompt_enhancer import PromptEnhancer
from dreamverse.prompt_safety import PromptSafetyFilter

import dreamverse.runtime as runtime
from dreamverse.routes.health import (
    router as internal_monitor_router, )
from dreamverse.routes.presets import (
    prompt_config_router,
    curated_presets_router,
)
from dreamverse.session.controller import SessionController


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

app.include_router(build_health_router(lambda: runtime.gpu_pool))
app.include_router(internal_monitor_router)
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


class LoraRequest(BaseModel):
    strength: float = 1.0
    styles: dict[str, float] = {}
    style: str = ""


@app.get("/lora/options")
async def lora_options() -> dict:
    if not DEVTOOLS_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    styles = _available_styles_for_active_model()
    has_base_lora = _resolve_lora_spec("omninft") is not None
    labels = {"none": "None"}
    labels.update({k: AVAILABLE_LORAS[k].get("label", k) for k in styles})
    return {
        "styles": ["none", *styles],
        "labels": labels,
        "has_base_lora": has_base_lora,
    }


@app.post("/lora")
async def apply_lora(request: LoraRequest) -> dict:
    if not DEVTOOLS_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    if runtime.gpu_pool is None:
        raise HTTPException(status_code=503, detail="GPU pool not ready")

    strength = max(0.0, min(1.0, float(request.strength)))
    allowed_styles = _available_styles_for_active_model()

    requested = dict(request.styles) if request.styles else {}
    if not requested and request.style and request.style.strip().lower() != "none":
        requested = {request.style: 1.0}

    styles_map: dict[str, float] = {}
    for name, intensity in requested.items():
        key = str(name).strip().lower()
        if key in ("", "none"):
            continue
        if key not in allowed_styles:
            raise HTTPException(status_code=400, detail=f"Unknown style for active model: {key}")
        value = max(0.0, min(1.0, float(intensity)))
        if value <= 0.0:
            continue
        styles_map[key] = value

    stack: list[tuple[str, float]] = []
    if _resolve_lora_spec("omninft") is not None:
        stack.append(("omninft", strength))
    for key, intensity in styles_map.items():
        stack.append((key, intensity))

    if not stack:
        raise HTTPException(status_code=400,
                            detail="Active model has no OmniNFT LoRA and no style selected; nothing to apply.")

    try:
        triggers = await runtime.gpu_pool.apply_lora_stack(stack)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "applied": True,
        "strength": strength,
        "styles": styles_map,
        "triggers": {
            key: AVAILABLE_LORAS[key]["trigger"]
            for key in styles_map
        },
        "gpus": {
            str(gpu_id): trigger
            for gpu_id, trigger in triggers.items()
        },
    }


# Serve an exported frontend bundle when present.
for static_dir in FRONTEND_STATIC_DIR_CANDIDATES:
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
        break


def cli() -> None:
    import argparse
    import uvicorn

    from dreamverse._deps import require_dreamverse_runtime_deps

    require_dreamverse_runtime_deps()

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()

    _install_heartbeat_log_filter()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    cli()
