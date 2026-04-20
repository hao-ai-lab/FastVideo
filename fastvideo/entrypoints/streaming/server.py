# SPDX-License-Identifier: Apache-2.0
"""Single-generator FastAPI + WebSocket streaming server.

Minimum-viable implementation for PR 7.5:

* one ``VideoGenerator`` for the process (GPU pool lands in PR 7.6)
* one ``/v1/stream`` WebSocket endpoint accepting a
  ``session_init_v2`` opening frame plus subsequent
  ``segment_prompt_source`` frames
* fMP4 output through :class:`FragmentedMP4Encoder`
* continuation state held by :class:`InMemorySessionStore`, exportable
  via a ``snapshot_state`` client frame

Prompt enhancement (PR 7.7), auxiliaries (PR 7.8), router (PR 7.9),
and the async-event pipeline (PR 7.10) are explicitly out of scope.
This server runs a synchronous ``generator.generate(request)`` under
``asyncio.to_thread`` until PR 7.10's ``generate_async`` lands.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any

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
    ContinuationStateSnapshot,
    ErrorMessage,
    GpuAssigned,
    Ltx2SegmentComplete,
    Ltx2SegmentStart,
    Ltx2StreamComplete,
    Ltx2StreamStart,
    MediaInit,
    MediaSegmentComplete,
    QueueStatus,
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

# ---------------------------------------------------------------------------
# Generator protocol (tiny abstraction so tests can inject a mock)
# ---------------------------------------------------------------------------


class _GeneratorProto:
    """Subset of :class:`fastvideo.VideoGenerator` the server calls.

    Implementations just need to return a mapping with ``"frames"``
    (list of RGB uint8 HxWx3 ndarrays) and ``"audio_sample_rate"``.
    The real ``VideoGenerator.generate`` satisfies this today; tests
    wire in a fake for CPU-only runs.
    """

    def generate(self, request: GenerationRequest) -> Any:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    """Wiring the FastAPI app hands to each request handler."""

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
            await _send_json(websocket, ErrorMessage(
                code="session_rejected",
                message=str(exc),
                retryable=False,
            ))
            await websocket.close(code=1013, reason="session_rejected")
            return

        try:
            await _handle_session(websocket, session, state)
        except WebSocketDisconnect:
            logger.info("session %s: client disconnected", session.id[:8])
        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("session %s: unhandled error", session.id[:8])
            session.state = SessionState.ERROR
        finally:
            state.sessions.close(session.id)

    app.state.server_state = state
    return app


def run_server(serve_config: ServeConfig, *, generator: _GeneratorProto | None = None) -> None:
    """Launch the streaming server.

    Boots a :class:`fastvideo.VideoGenerator` from
    ``serve_config.generator`` unless ``generator`` is provided, then
    serves ``build_app(...)`` via uvicorn.
    """
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


# ---------------------------------------------------------------------------
# Session handler
# ---------------------------------------------------------------------------


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
    await _send_json(
        websocket,
        GpuAssigned(
            gpu_id=0,  # single-generator skeleton — GPU pool arrives in PR 7.6
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
        await _send_json(
            websocket,
            ErrorMessage(
                code="invalid_message",
                message=f"opening frame failed validation: {exc}",
                retryable=False,
            ))
        await websocket.close(code=1003, reason="invalid_init")
        session.state = SessionState.REJECTED
        return None
    if not isinstance(parsed, SessionInitV2):
        await _send_json(
            websocket,
            ErrorMessage(
                code="invalid_message",
                message="first frame must be session_init_v2",
                retryable=False,
            ))
        await websocket.close(code=1003, reason="expected_session_init_v2")
        session.state = SessionState.REJECTED
        return None
    return parsed


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
        image = persist_session_init_image(init.initial_image)
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
            await _send_json(websocket, ErrorMessage(
                code="invalid_message",
                message=str(exc),
                retryable=True,
            ))
            continue

        if isinstance(parsed, SnapshotState):
            snap = state.session_store.snapshot(session.id)
            if snap is None:
                await _send_json(
                    websocket,
                    ErrorMessage(
                        code="internal_error",
                        message="no continuation state available for session",
                        retryable=False,
                    ))
                continue
            await _send_json(websocket, ContinuationStateSnapshot(state={"kind": snap.kind, "payload": snap.payload}, ))
            continue

        if isinstance(parsed, SegmentPromptSource):
            await _run_segment(websocket, session, state, parsed)
            continue

        # Settings toggles from the catalogue — minimum wiring for PR 7.5.
        if hasattr(parsed, "enabled"):
            _apply_toggle(session, parsed)
            continue

        # Unknown but valid type — silently ignore per the additive
        # evolution rule (documented in streaming.md).


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
        await _send_json(
            websocket, ErrorMessage(
                code="worker_failed",
                message=f"generator.generate failed: {exc}",
                retryable=True,
            ))
        session.state = SessionState.ERROR
        return
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    frames = _extract_frames(result)
    if not frames:
        await _send_json(websocket,
                         ErrorMessage(
                             code="worker_failed",
                             message="generator returned no frames",
                             retryable=True,
                         ))
        session.state = SessionState.ERROR
        return

    # Emit step_complete events for observability even without a real
    # step-level progress source (PR 7.10 replaces this with events
    # from generate_async).
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
    session.generated_segment_count += 1
    with contextlib.suppress(InvalidSessionTransition):
        session.transition(SessionState.ACTIVE)

    await _send_json(
        websocket,
        Ltx2SegmentComplete(
            segment_idx=segment_idx,
            generation_time_ms=elapsed_ms,
            e2e_latency_ms=elapsed_ms,
        ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    enabled = bool(getattr(message, "enabled", False))
    name = getattr(message, "type", "")
    if name == "enhancement_updated":
        session.enhancement_enabled = enabled
    elif name == "auto_extension_updated":
        session.auto_extension_enabled = enabled
    elif name == "loop_generation_updated":
        session.loop_generation_enabled = enabled
    elif name == "generation_paused_updated":
        session.generation_paused = enabled


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


__all__ = [
    "ServerState",
    "build_app",
    "run_server",
]
