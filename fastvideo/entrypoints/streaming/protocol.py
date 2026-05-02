# SPDX-License-Identifier: Apache-2.0
"""JSON WebSocket protocol schemas for the streaming server.

Every control message shares the envelope ``{"type": <str>, ...}``.
Pydantic models live here so the server can parse / validate incoming
frames and emit well-typed outgoing frames without hand-rolled dicts.

The message catalogue matches the contract in
``docs/design/server_contracts/streaming.md``; additions must land in
both places in the same PR.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Client → server
# ---------------------------------------------------------------------------


class SessionInitV2(BaseModel):
    """Opening frame the client sends after the WebSocket handshake."""

    model_config = ConfigDict(extra="allow")

    type: Literal["session_init_v2"]
    client_id: str | None = None
    preset: str | None = None
    preset_label: str | None = None
    curated_prompts: list[str] = Field(default_factory=list)
    initial_image: dict[str, Any] | None = None
    enhancement_enabled: bool = False
    auto_extension_enabled: bool = False
    loop_generation_enabled: bool = False
    single_clip_mode: bool = False
    stream_mode: Literal["av_fmp4", "legacy_jpeg"] = "av_fmp4"
    continuation_state: dict[str, Any] | None = None
    """Optional ``{kind, payload}`` dict; hydrated into
    :class:`fastvideo.api.ContinuationState` server-side."""


class SegmentPromptSource(BaseModel):
    """Request a new segment using a specific prompt."""

    type: Literal["segment_prompt_source"]
    prompt: str
    negative_prompt: str | None = None
    source: Literal["curated", "enhanced", "user", "auto_extension"] = "user"
    seed: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None


class SeedPromptsUpdated(BaseModel):
    type: Literal["seed_prompts_updated"]
    seed_prompts: list[str] = Field(default_factory=list)


class EnhancementUpdated(BaseModel):
    type: Literal["enhancement_updated"]
    enabled: bool


class AutoExtensionUpdated(BaseModel):
    type: Literal["auto_extension_updated"]
    enabled: bool


class LoopGenerationUpdated(BaseModel):
    type: Literal["loop_generation_updated"]
    enabled: bool


class GenerationPausedUpdated(BaseModel):
    type: Literal["generation_paused_updated"]
    paused: bool


class SnapshotState(BaseModel):
    """Request the current ``ContinuationState`` for export."""

    type: Literal["snapshot_state"]


ClientMessage = Annotated[
    Union[  # noqa: UP007 - Annotated requires Union for discriminator
        SessionInitV2,
        SegmentPromptSource,
        SeedPromptsUpdated,
        EnhancementUpdated,
        AutoExtensionUpdated,
        LoopGenerationUpdated,
        GenerationPausedUpdated,
        SnapshotState,
    ],
    Field(discriminator="type"),
]

# ---------------------------------------------------------------------------
# Server → client
# ---------------------------------------------------------------------------


class QueueStatus(BaseModel):
    type: Literal["queue_status"] = "queue_status"
    position: int
    queue_depth: int


class GpuAssigned(BaseModel):
    type: Literal["gpu_assigned"] = "gpu_assigned"
    gpu_id: int
    session_timeout: int


class Ltx2StreamStart(BaseModel):
    type: Literal["ltx2_stream_start"] = "ltx2_stream_start"
    preset: str | None = None
    width: int
    height: int
    fps: int
    num_frames: int


class Ltx2SegmentStart(BaseModel):
    type: Literal["ltx2_segment_start"] = "ltx2_segment_start"
    segment_idx: int
    prompt: str
    total_steps: int


class StepComplete(BaseModel):
    type: Literal["step_complete"] = "step_complete"
    segment_idx: int
    step: int
    total_steps: int
    stage: str = "denoise"


class MediaInit(BaseModel):
    """Descriptor for the fMP4 initialization segment that follows."""

    type: Literal["media_init"] = "media_init"
    segment_idx: int
    mime: str = "video/mp4; codecs=\"avc1.64001f, mp4a.40.2\""
    stream_id: str
    mode: Literal["av_fmp4"] = "av_fmp4"


class MediaSegmentComplete(BaseModel):
    type: Literal["media_segment_complete"] = "media_segment_complete"
    segment_idx: int
    stream_id: str
    chunks: int
    duration_ms: float | None = None
    pts_base_ms: float | None = None


class Ltx2SegmentComplete(BaseModel):
    type: Literal["ltx2_segment_complete"] = "ltx2_segment_complete"
    segment_idx: int
    generation_time_ms: float
    e2e_latency_ms: float | None = None


class Ltx2StreamComplete(BaseModel):
    type: Literal["ltx2_stream_complete"] = "ltx2_stream_complete"
    reason: Literal["segment_cap", "stop_requested", "error"] = "stop_requested"


class SessionTimeout(BaseModel):
    type: Literal["session_timeout"] = "session_timeout"
    timeout_seconds: int


class ContinuationStateSnapshot(BaseModel):
    type: Literal["continuation_state_snapshot"] = "continuation_state_snapshot"
    state: dict[str, Any]
    """``{kind, payload}`` dict matching
    :class:`fastvideo.api.ContinuationState`."""


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: Literal[
        "session_rejected",
        "invalid_message",
        "preset_mismatch",
        "gpu_unavailable",
        "worker_failed",
        "upstream_timeout",
        "internal_error",
    ] = "internal_error"
    message: str
    retryable: bool = False


ServerMessage = Union[  # noqa: UP007 - pydantic Union handling
    QueueStatus,
    GpuAssigned,
    Ltx2StreamStart,
    Ltx2SegmentStart,
    StepComplete,
    MediaInit,
    MediaSegmentComplete,
    Ltx2SegmentComplete,
    Ltx2StreamComplete,
    SessionTimeout,
    ContinuationStateSnapshot,
    ErrorMessage,
]


def parse_client_message(raw: dict[str, Any]) -> ClientMessage:
    """Parse an incoming WebSocket dict into a typed client message.

    Unknown ``type`` values raise :class:`pydantic.ValidationError`; the
    server handler turns that into an ``error`` frame with
    ``code="invalid_message"``.
    """
    from pydantic import TypeAdapter

    return TypeAdapter(ClientMessage).validate_python(raw)


__all__ = [
    "AutoExtensionUpdated",
    "ClientMessage",
    "ContinuationStateSnapshot",
    "EnhancementUpdated",
    "ErrorMessage",
    "GenerationPausedUpdated",
    "GpuAssigned",
    "Ltx2SegmentComplete",
    "Ltx2SegmentStart",
    "Ltx2StreamComplete",
    "Ltx2StreamStart",
    "LoopGenerationUpdated",
    "MediaInit",
    "MediaSegmentComplete",
    "QueueStatus",
    "SeedPromptsUpdated",
    "SegmentPromptSource",
    "ServerMessage",
    "SessionInitV2",
    "SessionTimeout",
    "SnapshotState",
    "StepComplete",
    "parse_client_message",
]
