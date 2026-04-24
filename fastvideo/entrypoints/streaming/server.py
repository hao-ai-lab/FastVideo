# SPDX-License-Identifier: Apache-2.0
"""Single-generator FastAPI + WebSocket streaming server."""
from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from fastvideo.api.schema import (
    ContinuationState,
    GenerationRequest,
    InputConfig,
    OutputConfig,
    SamplingConfig,
    ServeConfig,
)
from fastvideo.entrypoints.streaming.protocol import (
    AutoExtensionUpdated,
    ContinuationStateSnapshot,
    EnhancementUpdated,
    ErrorMessage,
    GenerationPausedUpdated,
    GpuAssigned,
    LoopGenerationUpdated,
    Ltx2SegmentComplete,
    Ltx2SegmentStart,
    Ltx2StreamComplete,
    Ltx2StreamStart,
    MediaInit,
    MediaSegmentComplete,
    QueueStatus,
    SeedPromptsUpdated,
    SegmentPromptSource,
    SessionInitV2,
    SnapshotState,
    StepComplete,
    parse_client_message,
)
from fastvideo.entrypoints.streaming.session import (
    InvalidSessionTransition,
    Session,
    SessionManager,
    SessionRejected,
    SessionState,
)
from fastvideo.entrypoints.streaming.session_init_image import (
    persist_session_init_image, )
from fastvideo.entrypoints.streaming.session_store import (
    InMemorySessionStore,
    SessionStore,
)
from fastvideo.entrypoints.streaming.stream import FragmentedMP4Encoder
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# RFC 6455 WebSocket close codes used by the server.
_WS_CLOSE_UNSUPPORTED_DATA = 1003
_WS_CLOSE_TRY_AGAIN_LATER = 1013


class _GeneratorProto(Protocol):
    """Subset of :class:`fastvideo.VideoGenerator` the server calls."""

    def generate(self, request: GenerationRequest) -> Any:
        ...


@dataclass
class ServerState:
    serve_config: ServeConfig
    generator: _GeneratorProto
    sessions: SessionManager
    session_store: SessionStore


def build_app(
    serve_config: ServeConfig,
    generator: _GeneratorProto,
    *,
    session_store: SessionStore | None = None,
) -> FastAPI:
    """Build the FastAPI app used by :func:`run_server`.

    Exposed so tests can drive the WebSocket endpoint in-process via
    ``starlette.testclient.TestClient(app).websocket_connect(...)``.
    """
    if serve_config.streaming is None:
        raise ValueError("ServeConfig.streaming must be set to launch the streaming "
                         "server; got None. Add a `streaming:` block to your serve config.")

    sessions = SessionManager(
        segment_cap=serve_config.streaming.generation_segment_cap,
        session_timeout_seconds=serve_config.streaming.session_timeout_seconds,
    )
    state = ServerState(
        serve_config=serve_config,
        generator=generator,
        sessions=sessions,
        session_store=session_store or InMemorySessionStore(),
    )

    app = FastAPI(title="FastVideo Streaming")

    @app.get("/health")
    async def _health() -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "sessions": len(state.sessions),
            "stream_mode": state.serve_config.streaming.stream_mode,
        })

    @app.websocket("/v1/stream")
    async def _stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            session = state.sessions.create()
        except SessionRejected as exc:
            await _send_error(websocket, "session_rejected", str(exc), retryable=False)
            await websocket.close(code=_WS_CLOSE_TRY_AGAIN_LATER, reason="session_rejected")
            return

        try:
            await _handle_session(websocket, session, state)
        except WebSocketDisconnect:
            logger.info("session %s: client disconnected", session.id[:8])
        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("session %s: unhandled error", session.id[:8])
            with contextlib.suppress(InvalidSessionTransition):
                session.transition(SessionState.ERROR)
        finally:
            _cleanup_session(session, state)

    app.state.server_state = state
    return app


def run_server(serve_config: ServeConfig, *, generator: _GeneratorProto | None = None) -> None:
    """Launch the streaming server.

    Boots a :class:`fastvideo.VideoGenerator` from
    ``serve_config.generator`` unless ``generator`` is provided, then
    serves ``build_app(...)`` via uvicorn.
    """
    if serve_config.streaming is None:
        raise ValueError("ServeConfig.streaming must be set to launch the streaming server; "
                         "got None. Add a `streaming:` block to your serve config.")

    import uvicorn

    if generator is None:
        from fastvideo import VideoGenerator  # lazy to avoid boot cost

        generator = VideoGenerator.from_pretrained(config=serve_config.generator)
    app = build_app(serve_config, generator)
    uvicorn.run(
        app,
        host=serve_config.server.host,
        port=serve_config.server.port,
    )


async def _handle_session(
    websocket: WebSocket,
    session: Session,
    state: ServerState,
) -> None:
    init = await _read_init_message(websocket, session)
    if init is None:
        return

    await _apply_session_init(session, init, state)
    await _send_json(websocket, QueueStatus(position=0, queue_depth=0))
    session.transition(SessionState.GPU_BINDING)
    await _send_json(websocket, GpuAssigned(
        gpu_id=0,
        session_timeout=state.sessions.session_timeout_seconds,
    ))
    session.transition(SessionState.ACTIVE)
    await _send_json(websocket, _build_stream_start(session, state))

    try:
        await _run_segment_loop(websocket, session, state)
    finally:
        with contextlib.suppress(RuntimeError):
            await _send_json(websocket, Ltx2StreamComplete(reason="stop_requested"))


async def _read_init_message(
    websocket: WebSocket,
    session: Session,
) -> SessionInitV2 | None:
    try:
        raw = await websocket.receive_json()
    except WebSocketDisconnect:
        return None
    try:
        parsed = parse_client_message(raw)
    except Exception as exc:
        await _reject_init(websocket, session, f"opening frame failed validation: {exc}", "invalid_init")
        return None
    if not isinstance(parsed, SessionInitV2):
        await _reject_init(websocket, session, "first frame must be session_init_v2", "expected_session_init_v2")
        return None
    return parsed


async def _reject_init(
    websocket: WebSocket,
    session: Session,
    message: str,
    close_reason: str,
) -> None:
    await _send_error(websocket, "invalid_message", message, retryable=False)
    await websocket.close(code=_WS_CLOSE_UNSUPPORTED_DATA, reason=close_reason)
    with contextlib.suppress(InvalidSessionTransition):
        session.transition(SessionState.REJECTED)


async def _apply_session_init(
    session: Session,
    init: SessionInitV2,
    state: ServerState,
) -> None:
    session.client_id = init.client_id
    session.preset = init.preset
    session.preset_label = init.preset_label
    session.curated_prompts = list(init.curated_prompts)
    session.enhancement_enabled = init.enhancement_enabled
    session.auto_extension_enabled = init.auto_extension_enabled
    session.loop_generation_enabled = init.loop_generation_enabled
    session.single_clip_mode = init.single_clip_mode
    session.stream_mode = init.stream_mode

    if init.initial_image is not None:
        # Decode + disk write off the event loop; payload is up to 32 MiB.
        image = await asyncio.to_thread(persist_session_init_image, init.initial_image)
        if image is not None:
            session.metadata["session_init_image"] = image.path

    if init.continuation_state is not None:
        session.continuation_state = _coerce_state(init.continuation_state)
        if session.continuation_state is not None:
            state.session_store.store(session.id, session.continuation_state)


async def _run_segment_loop(
    websocket: WebSocket,
    session: Session,
    state: ServerState,
) -> None:
    cap = state.sessions.segment_cap
    while True:
        if session.segment_cap_reached(cap):
            logger.info("session %s: segment cap (%d) reached", session.id[:8], cap)
            return

        try:
            raw = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        session.touch()

        try:
            parsed = parse_client_message(raw)
        except Exception as exc:
            await _send_error(websocket, "invalid_message", str(exc), retryable=True)
            continue

        if isinstance(parsed, SnapshotState):
            snap = state.session_store.snapshot(session.id)
            if snap is None:
                await _send_error(websocket,
                                  "internal_error",
                                  "no continuation state available for session",
                                  retryable=False)
                continue
            await _send_json(websocket, ContinuationStateSnapshot(state={"kind": snap.kind, "payload": snap.payload}, ))
            continue

        if isinstance(parsed, SegmentPromptSource):
            await _run_segment(websocket, session, state, parsed)
            continue

        # Silently ignore unknown-but-valid types (additive-evolution
        # rule in streaming.md).
        _apply_toggle(session, parsed)


async def _run_segment(
    websocket: WebSocket,
    session: Session,
    state: ServerState,
    message: SegmentPromptSource,
) -> None:
    request = _build_generation_request(session, message, state)
    segment_idx = session.segment_idx
    await _send_json(
        websocket,
        Ltx2SegmentStart(
            segment_idx=segment_idx,
            prompt=message.prompt,
            total_steps=request.sampling.num_inference_steps,
        ))

    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, state.generator.generate, request)
    except Exception as exc:
        logger.exception("session %s: generator failed", session.id[:8])
        await _send_error(websocket, "worker_failed", f"generator.generate failed: {exc}", retryable=True)
        with contextlib.suppress(InvalidSessionTransition):
            session.transition(SessionState.ERROR)
        return
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    frames = _extract_frames(result)
    if not frames:
        await _send_error(websocket, "worker_failed", "generator returned no frames", retryable=True)
        with contextlib.suppress(InvalidSessionTransition):
            session.transition(SessionState.ERROR)
        return

    # Synchronous generator call has no per-step hook; emit one
    # terminal StepComplete so observability wiring still sees the
    # segment finish.
    total = request.sampling.num_inference_steps
    await _send_json(websocket, StepComplete(
        segment_idx=segment_idx,
        step=total,
        total_steps=total,
        stage="denoise",
    ))

    encoder = FragmentedMP4Encoder(
        width=request.sampling.width,
        height=request.sampling.height,
        fps=request.sampling.fps,
        segment_idx=segment_idx,
    )
    chunks_relayed = 0
    async with encoder:
        init_sent = False
        async for chunk in encoder.encode(frames):
            if chunk.kind == "init":
                await _send_json(websocket, MediaInit(
                    segment_idx=segment_idx,
                    stream_id=chunk.stream_id,
                ))
                init_sent = True
            await websocket.send_bytes(chunk.data)
            if init_sent and chunk.kind == "media":
                chunks_relayed += 1

    await _send_json(
        websocket,
        MediaSegmentComplete(
            segment_idx=segment_idx,
            stream_id=encoder.stream_id,
            chunks=chunks_relayed,
            duration_ms=float(request.sampling.num_frames) / request.sampling.fps * 1000.0,
        ))

    new_state = _extract_state(result)
    if new_state is not None:
        session.continuation_state = new_state
        state.session_store.store(session.id, new_state)

    session.segment_idx += 1
    with contextlib.suppress(InvalidSessionTransition):
        session.transition(SessionState.ACTIVE)

    await _send_json(
        websocket,
        Ltx2SegmentComplete(
            segment_idx=segment_idx,
            generation_time_ms=elapsed_ms,
            e2e_latency_ms=elapsed_ms,
        ))


def _build_stream_start(
    session: Session,
    state: ServerState,
) -> Ltx2StreamStart:
    default = state.serve_config.default_request
    return Ltx2StreamStart(
        preset=session.preset,
        width=default.sampling.width,
        height=default.sampling.height,
        fps=default.sampling.fps,
        num_frames=default.sampling.num_frames,
    )


def _build_generation_request(
    session: Session,
    message: SegmentPromptSource,
    state: ServerState,
) -> GenerationRequest:
    # Start from the operator-pinned default_request to pick up the
    # preset-selected sampling knobs; override with per-message values.
    base = state.serve_config.default_request
    sampling_kwargs: dict[str, Any] = {
        "num_videos_per_prompt":
        base.sampling.num_videos_per_prompt,
        "seed":
        message.seed if message.seed is not None else base.sampling.seed,
        "num_frames":
        base.sampling.num_frames,
        "height":
        base.sampling.height,
        "width":
        base.sampling.width,
        "fps":
        base.sampling.fps,
        "num_inference_steps":
        (message.num_inference_steps if message.num_inference_steps is not None else base.sampling.num_inference_steps),
        "guidance_scale":
        (message.guidance_scale if message.guidance_scale is not None else base.sampling.guidance_scale),
    }
    request = GenerationRequest(
        prompt=message.prompt,
        negative_prompt=message.negative_prompt or base.negative_prompt,
        inputs=InputConfig(image_path=session.metadata.get("session_init_image"), ),
        sampling=SamplingConfig(**sampling_kwargs),
        output=OutputConfig(save_video=False, return_frames=True, return_state=True),
        state=session.continuation_state,
    )
    return request


def _coerce_state(raw: dict[str, Any]) -> ContinuationState | None:
    kind = raw.get("kind")
    payload = raw.get("payload")
    if not isinstance(kind, str) or not isinstance(payload, dict):
        return None
    return ContinuationState(kind=kind, payload=payload)


def _apply_toggle(session: Session, message: Any) -> None:
    if isinstance(message, EnhancementUpdated):
        session.enhancement_enabled = message.enabled
    elif isinstance(message, AutoExtensionUpdated):
        session.auto_extension_enabled = message.enabled
    elif isinstance(message, LoopGenerationUpdated):
        session.loop_generation_enabled = message.enabled
    elif isinstance(message, GenerationPausedUpdated):
        session.generation_paused = message.paused
    elif isinstance(message, SeedPromptsUpdated):
        session.curated_prompts = list(message.seed_prompts)


def _extract_frames(result: Any) -> list:
    if hasattr(result, "frames"):
        return list(result.frames or [])
    if isinstance(result, dict):
        return list(result.get("frames") or [])
    return []


def _extract_state(result: Any) -> ContinuationState | None:
    state = getattr(result, "state", None)
    if state is None and isinstance(result, dict):
        state = result.get("state")
    if isinstance(state, ContinuationState):
        return state
    if isinstance(state, dict):
        return _coerce_state(state)
    return None


async def _send_json(websocket: WebSocket, message: Any) -> None:
    payload = (message.model_dump(mode="json", exclude_none=True) if hasattr(message, "model_dump") else message)
    await websocket.send_json(payload)


async def _send_error(
    websocket: WebSocket,
    code: str,
    message: str,
    *,
    retryable: bool,
) -> None:
    await _send_json(
        websocket,
        ErrorMessage(code=code, message=message, retryable=retryable),
    )


def _cleanup_session(session: Session, state: ServerState) -> None:
    state.sessions.close(session.id)
    state.session_store.drop(session.id)
    init_image_path = session.metadata.get("session_init_image")
    if isinstance(init_image_path, str):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(init_image_path)


__all__ = [
    "ServerState",
    "build_app",
    "run_server",
]
