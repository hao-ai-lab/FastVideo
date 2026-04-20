# SPDX-License-Identifier: Apache-2.0
"""Router FastAPI entry point.

Exposes the same ``/v1/stream`` WebSocket path the backend servers do,
accepts a client, picks a healthy replica from the registry, and
proxies frames bidirectionally.

PR 7.9 ships the minimum-viable shape: explicit replica list, single
primary, JSON + binary passthrough in both directions, and a
``/status`` endpoint for operators. Sticky-session routing (so a
reconnect lands on the same backend) is left for a follow-up.
"""
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from fastvideo.entrypoints.streaming.router.config import RouterConfig
from fastvideo.entrypoints.streaming.router.registry import (
    ReplicaRegistry,
    run_health_check_loop,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _RouterState:
    config: RouterConfig
    registry: ReplicaRegistry
    stop_event: asyncio.Event
    health_task: asyncio.Task | None = None


def build_router_app(
    config: RouterConfig,
    *,
    registry: ReplicaRegistry | None = None,
) -> FastAPI:
    """Build the router FastAPI app.

    ``registry`` can be injected for tests; defaults to one built from
    ``config.replicas``.
    """
    registry = registry or ReplicaRegistry(config.replicas)
    state = _RouterState(
        config=config,
        registry=registry,
        stop_event=asyncio.Event(),
    )
    app = FastAPI(title="FastVideo Streaming Router")

    @app.on_event("startup")
    async def _startup() -> None:
        state.health_task = asyncio.create_task(
            run_health_check_loop(
                registry=state.registry,
                config=state.config,
                stop_event=state.stop_event,
            ))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        state.stop_event.set()
        if state.health_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await state.health_task

    @app.get("/status")
    async def _status() -> JSONResponse:
        return JSONResponse({
            "replicas": [{
                "url": r.url,
                "primary": r.primary,
                "status": r.health.status.value,
                "last_ok_at": r.health.last_ok_at,
                "last_latency_ms": r.health.last_latency_ms,
                "consecutive_failures": r.health.consecutive_failures,
            } for r in state.registry.all()],
        })

    @app.websocket("/v1/stream")
    async def _proxy(websocket: WebSocket) -> None:
        await websocket.accept()
        replica = state.registry.select()
        if replica is None:
            await websocket.send_json({
                "type": "error",
                "code": "gpu_unavailable",
                "message": "router: no healthy replica available",
                "retryable": True,
            })
            await websocket.close(code=1013, reason="no_healthy_replica")
            return

        ws_url = _websocket_url_for(replica.url)
        try:
            await _bridge_session(websocket, ws_url)
        except WebSocketDisconnect:
            logger.info("router: client disconnected")
        except Exception as exc:
            logger.exception("router: bridge failed: %s", exc)
            with contextlib.suppress(RuntimeError):
                await websocket.send_json({
                    "type": "error",
                    "code": "worker_failed",
                    "message": f"router bridge failed: {exc}",
                    "retryable": True,
                })
            with contextlib.suppress(RuntimeError):
                await websocket.close(code=1011)

    app.state.router_state = state
    return app


def run_router(config: RouterConfig) -> None:  # pragma: no cover - CLI
    import uvicorn

    app = build_router_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


async def _bridge_session(
    client_ws: WebSocket,
    backend_ws_url: str,
) -> None:
    """Connect to backend and shuttle messages in both directions.

    Uses ``websockets`` for the backend side; imported lazily to keep
    the router's import graph small for users who only want the server.
    """
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("router requires the `websockets` package for backend proxying") from exc

    async with websockets.connect(backend_ws_url + "/v1/stream") as backend_ws:
        await asyncio.gather(
            _forward_client_to_backend(client_ws, backend_ws),
            _forward_backend_to_client(backend_ws, client_ws),
        )


async def _forward_client_to_backend(client_ws: WebSocket, backend_ws) -> None:
    try:
        while True:
            msg = await client_ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "text" in msg and msg["text"] is not None:
                await backend_ws.send(msg["text"])
            elif "bytes" in msg and msg["bytes"] is not None:
                await backend_ws.send(msg["bytes"])
    finally:
        with contextlib.suppress(Exception):
            await backend_ws.close()


async def _forward_backend_to_client(backend_ws, client_ws: WebSocket) -> None:
    try:
        async for frame in backend_ws:
            if isinstance(frame, bytes):
                await client_ws.send_bytes(frame)
            else:
                await client_ws.send_text(frame)
    finally:
        with contextlib.suppress(Exception):
            await client_ws.close()


def _websocket_url_for(http_url: str) -> str:
    if http_url.startswith("https://"):
        return "wss://" + http_url[len("https://"):]
    if http_url.startswith("http://"):
        return "ws://" + http_url[len("http://"):]
    return http_url


__all__ = [
    "build_router_app",
    "run_router",
]
