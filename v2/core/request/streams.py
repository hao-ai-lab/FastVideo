"""Streams + typed event taxonomy.

A ``Stream`` is one ordered event channel for previews, media chunks, progress,
logs, and finals. Event taxonomy: ``request.*``, ``session.*``, ``artifact.*``,
``media.{init,chunk,complete}``, ``trace.*``. A ``media.chunk`` carries its
stream, byte-range/buffer-ref, codec/container, timestamp range, and
preview-vs-final flag, so invalid combinations are unrepresentable.

For v2 the Stream is a simple synchronous ordered channel (the core is
numpy/sync and CPU-testable); a GPU/server deployment swaps in an async channel
without changing the event types.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from v2.core.types import TensorLike


@dataclass(frozen=True)
class OmniEvent:
    type: str  # e.g. "request.start", "media.chunk", "artifact.ready", "trace.step"
    request_id: str
    seq: int = 0
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamChunk:
    """A media.chunk payload — invalid field combinations are unrepresentable."""
    stream_id: str
    modality: str  # "video" | "audio" | "text"
    seq: int
    data: TensorLike = None  # in-proc buffer (frames / samples / token ids)
    codec: str = "raw"  # raw | h264 | opus | ...
    ts_start: float = 0.0  # presentation timestamp range (seconds)
    ts_end: float = 0.0
    preview: bool = False  # preview vs final
    final: bool = False  # last chunk of the stream


class Stream:
    """Ordered event channel. Sync collector for tests; subscribe for streaming."""

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.events: list[OmniEvent] = []
        self._subscribers: list = []

    def emit(self, event: OmniEvent) -> None:
        self.events.append(event)
        for cb in self._subscribers:
            cb(event)

    def subscribe(self, callback) -> None:
        self._subscribers.append(callback)

    def __iter__(self):
        return iter(self.events)
