# SPDX-License-Identifier: Apache-2.0
"""Request body and SSE event types of ``POST /v1/streamv2/generate``."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoraSelection(BaseModel):
    """The adapter a request selects, and the strength to apply it at.

    ``id`` names one adapter of the server's allow-list, as ``GET /v1/streamv2/capabilities`` reports it.
    ``strength`` multiplies the published adapter delta; omitted, the adapter's default strength applies, and
    ``0`` returns the base model's output unchanged.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    strength: float | None = Field(default=None, ge=0.0)


class Reference(BaseModel):
    """One ordered reference: its media type and its base64 bytes."""

    model_config = ConfigDict(extra="forbid")

    # A media type of ``MiniMaxH3Reference``; the capabilities route lists the accepted ones.
    media_type: str = Field(min_length=1)
    # The reference file's bytes as base64 without a ``data:`` prefix.
    data: str


class GenerateRequest(BaseModel):
    """Body of one segment request; every generation input comes from the client."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    # List order is model order: the model numbers references per media type
    # (``<Picture N>``, ``<Video N>``, ...) in this order. Empty is text-to-video.
    references: list[Reference] = Field(default_factory=list)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    num_frames: int = Field(gt=0)
    seed: int | None = None
    return_last_frame: bool = False
    # Omitted, the request renders with the base model (or, after another request loaded an adapter, at strength 0,
    # which is the base model's output).
    lora: LoraSelection | None = None


class LastFrameEvent(BaseModel):
    """The final decoded frame as base64 PNG, sent before any video."""

    data: str


class VideoStartEvent(BaseModel):
    """The MIME type, with codecs, of the fragmented MP4 that the chunks form."""

    mime: str


class VideoChunkEvent(BaseModel):
    """One piece of the fragmented MP4 as base64; pieces concatenate in order."""

    data: str


class DoneEvent(BaseModel):
    """Millisecond timings of a completed segment."""

    timings: dict[str, float]


class ErrorEvent(BaseModel):
    """A failure; also the body of the HTTP 400 answer to an invalid request.

    ``invalid_request`` means the request itself is wrong. ``generation_failed``
    means generation or encoding failed for another reason.
    """

    code: Literal["invalid_request", "generation_failed"]
    message: str


StreamEvent = LastFrameEvent | VideoStartEvent | VideoChunkEvent | DoneEvent | ErrorEvent

_EVENT_NAMES: dict[type[BaseModel], str] = {
    LastFrameEvent: "last_frame",
    VideoStartEvent: "video_start",
    VideoChunkEvent: "video_chunk",
    DoneEvent: "done",
    ErrorEvent: "error",
}


def format_sse(event: StreamEvent) -> str:
    """Serialize one event as an SSE frame with its name and one-line JSON data."""
    return f"event: {_EVENT_NAMES[type(event)]}\ndata: {event.model_dump_json()}\n\n"
