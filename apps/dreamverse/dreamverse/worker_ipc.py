# pyright: reportMissingTypeArgument=false
"""Typed worker IPC events: replaces the fat ``Response`` dataclass.

Each message kind from GPU worker → main process is its own small
dataclass; ``WorkerEvent`` is the union of them.  Consumers dispatch
via ``match``/``case`` or ``isinstance`` — mirroring the existing
``StreamEvent`` pattern in ``av_streaming.py``.

Invalid states are unrepresentable: a ``MediaChunk`` simply has no
``frames`` field, a ``StepComplete`` has no ``chunk_offset`` field,
and a ``JoinAck`` can't accidentally default to ``kind="step_result"``
because there is no ``kind`` string.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- User-scoped events (carry user_id) ------------------------------------


@dataclass(frozen=True)
class StepComplete:
    """Generation step finished.  Frames/audio were already sent via ``MediaChunk``."""
    user_id: str
    segment_idx: int
    timings: dict[str, float]


@dataclass(frozen=True)
class WorkerError:
    """Any worker failure.  ``user_id`` is None for system-scoped failures."""
    user_id: str | None
    message: str


@dataclass(frozen=True)
class JoinAck:
    user_id: str


@dataclass(frozen=True)
class LeaveAck:
    user_id: str


@dataclass(frozen=True)
class ReloadAck:
    user_id: str


@dataclass(frozen=True)
class WarmupComplete:
    user_id: str | None
    timings: dict[str, float]


# ---- Streaming events (carry user_id + segment_idx for routing) ------------


@dataclass(frozen=True)
class MediaInit:
    user_id: str
    segment_idx: int
    stream_id: str
    mime: str
    uses_shared_buffer: bool


@dataclass(frozen=True)
class MediaChunk:
    """One fMP4 chunk.

    Either ``chunk`` (raw bytes) is set, or both ``chunk_offset`` and
    ``chunk_length`` are set (read from the shared buffer).  The
    invariant is enforced in ``__post_init__``.
    """
    user_id: str
    segment_idx: int
    stream_id: str
    chunk: bytes | None = None
    chunk_offset: int | None = None
    chunk_length: int | None = None
    uses_shared_buffer: bool = False

    def __post_init__(self) -> None:
        has_bytes = self.chunk is not None
        has_offset = (self.chunk_offset is not None and self.chunk_length is not None)
        if has_bytes == has_offset:
            raise ValueError("MediaChunk must carry either chunk bytes or "
                             "(chunk_offset + chunk_length), not both or neither")


@dataclass(frozen=True)
class MediaComplete:
    user_id: str
    segment_idx: int
    stream_id: str
    chunks: int


# ---- System events (no user_id) --------------------------------------------


@dataclass(frozen=True)
class InitAck:
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class ShutdownAck:
    pass


WorkerEvent = (StepComplete
               | WorkerError
               | JoinAck
               | LeaveAck
               | ReloadAck
               | WarmupComplete
               | MediaInit
               | MediaChunk
               | MediaComplete
               | InitAck
               | ShutdownAck)

# ---- Command payloads (main process → worker) ------------------------------
#
# The envelope (``Command`` + ``CommandType``) lives in ``gpu_pool.py``
# alongside the enum; only the typed payloads live here so they sit
# next to the response-side types.  Commands without a payload
# (``INIT``, ``SHUTDOWN``, ``USER_JOIN``, ``USER_LEAVE``) leave
# ``Command.payload`` as ``None``.


@dataclass(frozen=True)
class UserStepPayload:
    prompt: str
    segment_idx: int
    image_path: str | None
    reset_conditioning: bool


@dataclass(frozen=True)
class WarmupPayload:
    prompt: str


@dataclass(frozen=True)
class ReloadModelPayload:
    # ``model_config`` stays a dict because ``MODEL_REGISTRY`` values
    # are dicts shaped by the external config module; typing that dict
    # is a separate refactor.
    model_config: dict


CommandPayload = UserStepPayload | WarmupPayload | ReloadModelPayload
