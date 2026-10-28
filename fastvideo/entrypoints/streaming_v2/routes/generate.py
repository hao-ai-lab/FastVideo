# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/streamv2/generate``: generate one segment and stream it back as SSE."""
from __future__ import annotations

import asyncio
import base64
import io
import shutil
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image

from fastvideo.api.compat import explicit_request_updates, legacy_generate_call_to_request
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.streaming_v2.messages import (
    DoneEvent,
    ErrorEvent,
    GenerateRequest,
    LastFrameEvent,
    Reference,
    VideoChunkEvent,
    VideoStartEvent,
    format_sse,
)
from fastvideo.entrypoints.streaming_v2.video_stream import encode_fmp4, video_mime
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference, validate_references

if TYPE_CHECKING:
    from fastvideo.entrypoints.streaming_v2.server import ServerState

logger = init_logger(__name__)
router = APIRouter()


@router.post("/v1/streamv2/generate")
async def generate_segment(raw_request: Request) -> Response:
    """Validate the request, then stream the segment; an invalid request gets HTTP 400 before any event."""
    received = time.perf_counter()
    state: ServerState = raw_request.app.state.streaming_v2
    try:
        body = GenerateRequest.model_validate(await raw_request.json())
        references = _validate_request_references(body.references)
    except ValueError as error:  # malformed JSON, schema violations, bad references
        error_body = ErrorEvent(code="invalid_request", message=str(error))
        return JSONResponse(status_code=400, content=error_body.model_dump())
    return StreamingResponse(
        stream_segment(state, body, references, received),
        media_type="text/event-stream",
    )


def _validate_request_references(references: list[Reference]) -> list[MiniMaxH3Reference]:
    """Decode and check the request's references; a request without any is text-to-video and is not checked."""
    if not references:
        return []
    return validate_references([_decode_reference(index, reference) for index, reference in enumerate(references)])


def build_generation_request(
    body: GenerateRequest,
    references: list[MiniMaxH3Reference],
    default_request: GenerationRequest | None,
) -> GenerationRequest:
    """Merge the operator's explicit defaults with the body; the body's prompt, size, frames, seed, and references win.

    Frames and audio are returned in memory, not saved.
    """
    kwargs: dict[str, Any] = explicit_request_updates(default_request) if default_request is not None else {}
    kwargs.update(width=body.width, height=body.height, num_frames=body.num_frames)
    if body.seed is not None:
        kwargs["seed"] = body.seed
    kwargs["references"] = references
    kwargs.update(save_video=False, return_frames=True)
    return legacy_generate_call_to_request(body.prompt, None, legacy_kwargs=kwargs)


async def stream_segment(
    state: ServerState,
    body: GenerateRequest,
    references: list[MiniMaxH3Reference],
    received: float,
) -> AsyncIterator[str]:
    """Generate one segment and yield its SSE frames: optional ``last_frame``, ``video_start``, chunks, ``done``.

    The references' bytes go to a fresh directory once the stream runs; the generation task removes it when
    generation finishes, even if the client has left. A ``ValueError`` is the request's fault and ends the stream
    with ``invalid_request``; any other failure ends it with ``generation_failed``.
    """
    try:
        generation_request, reference_dir = _prepare_request(body, references, state.default_request)
        result, generation_ms = await state.generate(generation_request, reference_dir)
    except ValueError as error:
        logger.warning("streaming_v2 rejected a request: %s", error)
        yield format_sse(ErrorEvent(code="invalid_request", message=str(error)))
        return
    except Exception as error:
        logger.exception("streaming_v2 generation failed")
        yield format_sse(ErrorEvent(code="generation_failed", message=str(error)))
        return

    try:
        frames, audio, sample_rate = _unpack_result(result)
        if body.return_last_frame:
            png = await asyncio.to_thread(_encode_png, frames[-1])
            yield format_sse(LastFrameEvent(data=base64.b64encode(png).decode("ascii")))
        yield format_sse(VideoStartEvent(mime=video_mime(has_audio=audio is not None)))
        async for chunk in encode_fmp4(frames, generation_request.sampling.fps, audio, sample_rate):
            yield format_sse(VideoChunkEvent(data=base64.b64encode(chunk).decode("ascii")))
        timings = {"generation_ms": generation_ms, "e2e_latency_ms": (time.perf_counter() - received) * 1000.0}
        yield format_sse(DoneEvent(timings=timings))
    except Exception as error:
        logger.exception("streaming_v2 failed to stream a generated segment")
        yield format_sse(ErrorEvent(code="generation_failed", message=str(error)))


def _prepare_request(
    body: GenerateRequest,
    references: list[MiniMaxH3Reference],
    default_request: GenerationRequest | None,
) -> tuple[GenerationRequest, str | None]:
    """Build the generation request over reference files in a fresh directory, which is ``None`` without references.

    The caller hands the directory to ``ServerState.generate``, which removes it once generation ends.
    """
    if not references:
        return build_generation_request(body, [], default_request), None
    reference_dir = tempfile.mkdtemp(prefix="streamv2-references-")
    try:
        files = _write_reference_files(references, reference_dir)
        return build_generation_request(body, files, default_request), reference_dir
    except Exception:
        shutil.rmtree(reference_dir, ignore_errors=True)
        raise


def _decode_reference(index: int, reference: Reference) -> MiniMaxH3Reference:
    """Turn one request reference into a model reference over its bytes.

    Raises ``ValueError`` when ``data`` is not base64 or ``MiniMaxH3Reference``
    refuses the media type. The model decodes the bytes later, by media type.
    """
    try:
        data = base64.b64decode(reference.data, validate=True)
    except ValueError as error:  # binascii.Error is a ValueError
        raise ValueError(f"references[{index}].data is not base64: {error}") from error
    return MiniMaxH3Reference(source=data, media_type=reference.media_type)


def _write_reference_files(references: list[MiniMaxH3Reference], reference_dir: str) -> list[MiniMaxH3Reference]:
    """Write each reference's bytes to its own file in ``reference_dir``; the returned references read those files."""
    written = []
    for index, reference in enumerate(references):
        path = Path(reference_dir) / str(index)
        path.write_bytes(reference.source)
        written.append(replace(reference, source=str(path)))
    return written


def _unpack_result(result: Any) -> tuple[list[np.ndarray], Any | None, int | None]:
    """Take the decoded frames, audio, and audio sample rate of the request's single result."""
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"Expected one generation result, got {len(result)}.")
        result = result[0]
    if not result.frames:
        raise RuntimeError("The generator returned no frames.")
    return list(result.frames), result.audio, result.audio_sample_rate


def _encode_png(frame: np.ndarray) -> bytes:
    """Encode one HxWx3 uint8 RGB frame as PNG, favoring speed over size."""
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
