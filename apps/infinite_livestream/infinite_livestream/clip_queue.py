"""The clip queues: what the stream holds, in order, at each stage.

A clip is enqueued (generation queue), built (playout queue), then consumed by
playing. Both stages are the same bounded, ordered, position-addressable
container; `engine.py` owns when an entry crosses between them.

Pure bookkeeping, so it is testable without a GPU.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from . import clip_plan


@dataclass
class ClipEntry:
    """One clip, from request to built payload.

    Everything but `video`/`audio` is frozen at enqueue time; those two arrive
    when the build completes, and `ready` is derived from their presence.
    """

    clip_id: str
    prompt: str
    metadata: str
    frames: int
    seed: int
    # Set while a build for this entry is in flight, so the scheduler never
    # submits the same entry twice.
    building: bool = False
    # The built payload: decoded RGB frames and the wire-ready waveform.
    video: list[Any] | None = None
    audio: Any = None

    @property
    def ready(self) -> bool:
        return self.video is not None

    @property
    def seconds(self) -> float:
        return clip_plan.seconds_for_frames(self.frames)

    def snapshot(self) -> dict[str, Any]:
        """The clip as every message that references it carries it.

        Whole rather than an id, so a listener never has to join against an
        earlier message; a plain mapping, so it is JSON-serialisable for the
        websocket.
        """
        return {
            "clip_id": self.clip_id,
            "prompt": self.prompt,
            "metadata": self.metadata,
            "frames": self.frames,
            "seconds": round(self.seconds, 3),
            "seed": self.seed,
            "ready": self.ready,
        }


def new_entry(*, prompt: str, metadata: str, frames: int, seed: int) -> ClipEntry:
    """Mint one entry with a fresh UUID."""
    return ClipEntry(
        clip_id=str(uuid.uuid4()),
        prompt=prompt,
        metadata=metadata,
        frames=frames,
        seed=seed,
    )


@dataclass
class ClipQueue:
    """A bounded, ordered, position-addressable queue of `ClipEntry`.

    One container serves both stages. Positions are explicit and nothing
    reorders on its own. For the playout queue every entry holds a fully
    decoded clip in host memory, so `capacity` is also the memory budget.
    """

    capacity: int
    _entries: list[ClipEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"queue capacity must be positive, got {self.capacity}")

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry: ClipEntry) -> bool:
        return any(existing is entry for existing in self._entries)

    @property
    def full(self) -> bool:
        return len(self._entries) >= self.capacity

    def add(self, entry: ClipEntry, position: int | None = None) -> int:
        """Insert at `position` (None appends, otherwise clamped) and return the index."""
        if self.full:
            raise ValueError(f"the queue is full ({self.capacity} clips)")
        index = (len(self._entries) if position is None else max(0, min(int(position), len(self._entries))))
        self._entries.insert(index, entry)
        return index

    def move(self, entry: ClipEntry, position: int) -> int:
        """Reposition `entry` and return the index it landed at, clamped."""
        if entry not in self:
            raise ValueError("the clip is not in this queue")
        self._entries = [existing for existing in self._entries if existing is not entry]
        index = max(0, min(int(position), len(self._entries)))
        self._entries.insert(index, entry)
        return index

    def get(self, clip_id: str) -> ClipEntry | None:
        for entry in self._entries:
            if entry.clip_id == clip_id:
                return entry
        return None

    def head(self) -> ClipEntry | None:
        return self._entries[0] if self._entries else None

    def next_to_build(self) -> ClipEntry | None:
        """The front-most entry no build is already running for."""
        for entry in self._entries:
            if not entry.building:
                return entry
        return None

    def remove(self, entry: ClipEntry) -> None:
        self._entries = [existing for existing in self._entries if existing is not entry]

    def clear(self) -> int:
        """Drop every entry, built payloads included, and return how many."""
        cleared = len(self._entries)
        self._entries = []
        return cleared

    def snapshot(self) -> list[dict[str, Any]]:
        return [entry.snapshot() for entry in self._entries]


__all__ = ["ClipEntry", "ClipQueue", "new_entry"]
