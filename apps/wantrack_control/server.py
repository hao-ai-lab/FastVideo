"""FastAPI/WebSocket server for causal WanTrack control."""

from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
import io
import os
from pathlib import Path
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from apps.wantrack_control.media import FMP4BlockWriter

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _decode_image(value: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("prepare.image must be a non-empty base64 string")
    encoded = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("prepare.image is not valid base64") from exc
    if not data:
        raise ValueError("prepare.image decoded to no bytes")
    return data


def _encode_image(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _block_value(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


class _RuntimeProvider:

    def __init__(self, runtime: Any | None) -> None:
        self._runtime = runtime
        self._lock = asyncio.Lock()

    async def get(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        async with self._lock:
            if self._runtime is None:
                model_dir = os.getenv("WANTRACK_MODEL_DIR", "").strip()
                yaml_path = os.getenv("WANTRACK_YAML_PATH", "").strip()
                if not model_dir or not yaml_path:
                    raise RuntimeError("Set WANTRACK_MODEL_DIR and WANTRACK_YAML_PATH before "
                                       "preparing a session")
                from fastvideo.train.models.wantrack.runtime import (
                    WanTrackInferenceRuntime, )

                self._runtime = await asyncio.to_thread(
                    WanTrackInferenceRuntime.from_export,
                    model_dir,
                    yaml_path,
                )
        return self._runtime


def create_app(
    runtime: Any | None = None,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    writer_factory: Any = FMP4BlockWriter,
) -> FastAPI:
    app = FastAPI(title="Causal WanTrack Control")
    app.state.runtime_provider = _RuntimeProvider(runtime)
    app.state.active_generation = asyncio.Lock()
    app.state.downloads = {}
    if output_dir is None:
        resolved_output_dir: str | os.PathLike[str] = os.getenv(
            "WANTRACK_OUTPUT_DIR",
            str(Path(tempfile.gettempdir()) / "wantrack_control"),
        )
    else:
        resolved_output_dir = output_dir
    app.state.output_dir = Path(resolved_output_dir)
    app.state.writer_factory = writer_factory

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "active": app.state.active_generation.locked(),
        }

    @app.get("/downloads/{download_id}")
    async def download(download_id: str) -> FileResponse:
        path = app.state.downloads.get(download_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="download not found")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename="wantrack_control.mp4",
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        send_lock = asyncio.Lock()
        prepared: Any | None = None
        session: Any | None = None
        generation_task: asyncio.Task[None] | None = None
        stop_event = asyncio.Event()
        owns_generation_lock = False
        connected = True

        async def send_json(payload: dict[str, Any]) -> None:
            nonlocal connected
            if not connected:
                return
            async with send_lock:
                try:
                    await websocket.send_json(payload)
                except Exception:
                    connected = False
                    raise

        async def send_bytes(payload: bytes) -> None:
            nonlocal connected
            if not connected:
                return
            async with send_lock:
                try:
                    await websocket.send_bytes(payload)
                except Exception:
                    connected = False
                    raise

        async def run_generation(runtime_value: Any) -> None:
            nonlocal owns_generation_lock, connected
            writer = app.state.writer_factory(
                app.state.output_dir,
                fps=float(getattr(runtime_value, "fps", 16.0)),
            )
            init_sent = False
            last_applied_revision = 0
            terminal_error: str | None = None
            try:
                while not stop_event.is_set():
                    block_index = int(getattr(session, "block_index", 0))
                    await send_json({
                        "type": "block_started",
                        "block_index": block_index,
                    })
                    block = await asyncio.to_thread(session.generate_next_block)
                    applied_revision = int(_block_value(block, "applied_revision", 0))
                    if applied_revision > last_applied_revision:
                        last_applied_revision = applied_revision
                        await send_json({
                            "type": "control_applied",
                            "revision": applied_revision,
                            "block_index": int(_block_value(block, "block_index", block_index)),
                            "radius": float(_block_value(block, "radius", 0.0)),
                            "active_handle_ids": list(_block_value(block, "active_handle_ids", ())),
                        })
                    frames = _block_value(block, "pixel_frames")
                    encoded = await asyncio.to_thread(
                        writer.encode_block,
                        frames,
                        int(_block_value(block, "block_index", block_index)),
                    )
                    if not init_sent:
                        await send_json({
                            "type": "media_init",
                            "mime": encoded.mime,
                        })
                        await send_bytes(encoded.init_bytes)
                        init_sent = True
                    await send_bytes(encoded.media_bytes)
                    await send_json({
                        "type": "media_segment_complete",
                        "block_index": encoded.block_index,
                        "bytes": len(encoded.media_bytes),
                    })
            except Exception as exc:
                terminal_error = str(exc) or type(exc).__name__
            finally:
                if session is not None:
                    with suppress(Exception):
                        await asyncio.to_thread(
                            session.close,
                            "error" if terminal_error else ("disconnect" if not connected else "stop"),
                        )
                final_path = await asyncio.to_thread(writer.finalize)
                download_url = None
                if final_path is not None:
                    app.state.downloads[writer.session_id] = final_path
                    download_url = f"/downloads/{writer.session_id}"
                if owns_generation_lock:
                    app.state.active_generation.release()
                    owns_generation_lock = False
                if terminal_error:
                    if connected:
                        with suppress(Exception):
                            await send_json({
                                "type": "error",
                                "message": terminal_error,
                                "download_url": download_url,
                            })
                elif connected:
                    with suppress(Exception):
                        await send_json({
                            "type": "stream_complete",
                            "blocks": len(writer.block_paths),
                            "download_url": download_url,
                        })

        async def handle_message(message: dict[str, Any]) -> None:
            nonlocal prepared, session, generation_task
            nonlocal owns_generation_lock
            message_type = str(message.get("type", "")).strip()
            if message_type == "prepare":
                if generation_task is not None:
                    raise ValueError("prepare is unavailable during generation")
                runtime_value = await app.state.runtime_provider.get()
                image_bytes = _decode_image(message.get("image", ""))
                prompt = str(message.get("prompt", "") or "")
                prepared = await asyncio.to_thread(
                    runtime_value.prepare,
                    image_bytes,
                    prompt,
                )
                processed_image = getattr(prepared, "image", None)
                if not isinstance(processed_image, Image.Image):
                    processed_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                await send_json({
                    "type": "prepared",
                    "image": _encode_image(processed_image),
                    "width": processed_image.width,
                    "height": processed_image.height,
                    "fps": float(getattr(runtime_value, "fps", 16.0)),
                    "chunk_size": int(getattr(runtime_value, "chunk_size", 3)),
                })
                return

            if message_type == "start":
                if prepared is None:
                    raise ValueError("prepare must complete before start")
                if generation_task is not None:
                    raise ValueError("session is already generating")
                handles = message.get("handles")
                if not isinstance(handles, list) or not handles:
                    raise ValueError("start requires at least one handle")
                if app.state.active_generation.locked():
                    raise RuntimeError("another WanTrack session is already generating")
                await app.state.active_generation.acquire()
                owns_generation_lock = True
                runtime_value = await app.state.runtime_provider.get()
                session = runtime_value.create_session()
                sampling = {
                    "seed": int(message.get("seed", 0)),
                    "steps": int(message.get("steps", 30)),
                    "text_guidance": float(message.get("text_guidance", 3.0)),
                    "motion_guidance": float(message.get("motion_guidance", 1.5)),
                }
                try:
                    await asyncio.to_thread(
                        session.start,
                        prepared,
                        getattr(prepared, "prompt", str(message.get("prompt", "") or "")),
                        handles,
                        sampling,
                        radius=float(message.get("radius", 0.15)),
                    )
                except Exception:
                    app.state.active_generation.release()
                    owns_generation_lock = False
                    raise
                await send_json({
                    "type": "session_started",
                    "fps": float(getattr(runtime_value, "fps", 16.0)),
                    "chunk_size": int(getattr(runtime_value, "chunk_size", 3)),
                })
                generation_task = asyncio.create_task(run_generation(runtime_value))
                return

            if message_type == "control_update":
                if session is None:
                    raise ValueError("control_update requires a running session")
                revision = int(message.get("revision", 0))
                accepted = await asyncio.to_thread(
                    session.apply_control_revision,
                    revision,
                    samples=message.get("samples"),
                    add=message.get("add"),
                    remove=message.get("remove"),
                    handles=message.get("handles"),
                    radius=message.get("radius"),
                )
                if not accepted:
                    await send_json({
                        "type": "control_applied",
                        "revision": revision,
                        "status": "ignored_stale",
                    })
                return

            if message_type == "stop":
                if generation_task is None:
                    raise ValueError("stop requires a running session")
                stop_event.set()
                return
            raise ValueError(f"unknown client message type: {message_type!r}")

        try:
            while True:
                receive_task = asyncio.create_task(websocket.receive_json())
                waiters: set[asyncio.Task[Any]] = {receive_task}
                if generation_task is not None:
                    waiters.add(generation_task)
                done, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if generation_task is not None and generation_task in done:
                    receive_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await receive_task
                    await generation_task
                    return
                message = await receive_task
                try:
                    await handle_message(message)
                except Exception as exc:
                    await send_json({
                        "type": "error",
                        "message": str(exc) or type(exc).__name__,
                    })
                    if generation_task is not None:
                        stop_event.set()
        except WebSocketDisconnect:
            connected = False
        except Exception:
            connected = False
        finally:
            stop_event.set()
            if generation_task is not None:
                with suppress(Exception):
                    await generation_task
            elif owns_generation_lock:
                app.state.active_generation.release()

    return app


app = create_app()
