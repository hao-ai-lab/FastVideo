"""The engine: generation, playout, and the state every other module reads.

    director ──enqueue/pop/move──▶  Engine  ──frames+audio──▶  Pacer ──▶ sink
                                      │
                                      └──state_update / queue_update / clip_*
                                         ──▶ listeners (webapp, director)

The generator and the broadcast share a process, so a built clip is handed to
the pacer as the arrays it already is. There is no encode, no transport and
therefore nothing that can shed video frames while audio flows on -- which is
how a picture drifts behind its own soundtrack.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from . import clip_plan
from .backend import ClipJob, FastH3Backend
from .clip_queue import ClipEntry, ClipQueue, new_entry
from .config import Config, ModelConfig, require_weights, resolve_model_path
from .metadata import clip_view, encode_id3
from .pacer import Pacer

logger = logging.getLogger(__name__)

# Fixed by the checkpoint and the backend's resample; the canvas is not.
MODEL_FPS = clip_plan.FPS
MODEL_SAMPLE_RATE = 48_000

POLL_SECONDS = 0.05

# Frames handed to the pacer per step. Small keeps its buffers near-empty,
# which is the condition its A/V pairing depends on.
EMIT_FRAMES = 4


class Engine:
    """Own the model, the queues and the playout."""

    def __init__(self, config: Config, model_config: ModelConfig) -> None:
        self._config = config
        self._model = model_config
        self._pacer: Pacer | None = None
        self._listeners: list[Callable[[str, dict], None]] = []

        model_path = resolve_model_path(model_config, config.weights_path)
        require_weights(config.weights_path, model_path)
        self.backend = FastH3Backend(model_config, model_path)

        self._generation = ClipQueue(capacity=model_config.generation_queue_size)
        self._playout = ClipQueue(capacity=model_config.queue_size)
        # The build in flight: its entry, its job handle, and when it was
        # submitted (monotonic), so readiness latency is a measured number.
        self._build: tuple[ClipEntry, ClipJob, float] | None = None
        self._playing: ClipEntry | None = None

        self._seed = model_config.seed
        self._clips_played = 0
        self._frames_sent = 0
        self._seconds_sent = 0.0

        self._ready = asyncio.Event()
        # Mirrors of what listeners were last told, so a late subscriber (the
        # web app builds its own mirror from these) reads the same values.
        self.state: dict[str, Any] = self._snapshot()
        self.generation_clips: list[dict] = []
        self.playout_clips: list[dict] = []

    # ---------------------------------------------------------------- wiring

    def attach_pacer(self, pacer: Pacer) -> None:
        """Point the media path at the pacer."""
        self._pacer = pacer

    def add_listener(self, listener: Callable[[str, dict], None]) -> None:
        """Register for every message as `(kind, data)`. Must not raise."""
        self._listeners.append(listener)

    # ----------------------------------------------------------- state mirror

    @property
    def min_seconds(self) -> float:
        return clip_plan.MIN_SECONDS_PUBLISHED

    @property
    def max_seconds(self) -> float:
        return clip_plan.MAX_SECONDS_PUBLISHED

    @property
    def generation_queued(self) -> int:
        return len(self._generation)

    @property
    def generation_capacity(self) -> int:
        return self._generation.capacity

    @property
    def playout_queued(self) -> int:
        return len(self._playout)

    @property
    def playout_capacity(self) -> int:
        return self._playout.capacity

    @property
    def canvas(self) -> tuple[int, int]:
        """(width, height) this deployment generates at."""
        height, width = clip_plan.canvas_for_choice(self._model.aspect)
        return width, height

    @property
    def connected(self) -> bool:
        """Whether the model is loaded and commands would take effect."""
        return self._ready.is_set()

    def _canvas_hw(self) -> tuple[int, int]:
        return clip_plan.canvas_for_choice(self._model.aspect)

    def _snapshot(self) -> dict[str, Any]:
        """Everything an observer can see, in one mapping.

        The single source, so a joining viewer's greeting and everyone else's
        broadcast can never disagree.
        """
        height, width = self._canvas_hw()
        return {
            "width": width,
            "height": height,
            "playing": self._playing is not None,
            "generation_queued": len(self._generation),
            "generation_capacity": self._generation.capacity,
            "playout_queued": len(self._playout),
            "playout_capacity": self._playout.capacity,
            "clips_played": self._clips_played,
        }

    # -------------------------------------------------------------- messaging

    def _emit(self, kind: str, data: dict) -> None:
        """Fan one message out to every listener.

        Synchronous and non-throwing: these are in-process callbacks, and a
        broken listener must not take generation down with it.
        """
        for listener in self._listeners:
            try:
                listener(kind, data)
            except Exception:  # noqa: BLE001 -- a listener cannot break the engine
                logger.exception("[engine] listener failed on %s", kind)

    def _send_state(self) -> None:
        self.state = self._snapshot()
        self._emit("state_update", self.state)

    def _send_queue(self) -> None:
        self.generation_clips = self._generation.snapshot()
        self.playout_clips = self._playout.snapshot()
        self._emit(
            "queue_update",
            {
                "generation": self.generation_clips,
                "playout": self.playout_clips
            },
        )

    def _refuse(self, command: str, reason: str) -> None:
        logger.warning("[engine] %s refused: %s", command, reason)
        self._emit("command_error", {"command": command, "reason": reason})

    # --------------------------------------------------------------- commands

    async def send_command(self, command: str, data: dict) -> Any:
        """Dispatch one command, once the engine is up.

        Awaiting readiness (rather than failing) is what lets the director
        start before the model has finished loading: its first enqueue simply
        lands when the engine is ready for it. A ``None`` reply means the
        command was refused, and `command_error` carried the reason.
        """
        await self._ready.wait()
        handler = {
            "enqueue": self._enqueue,
            "pop": self._pop,
            "move": self._move,
        }.get(command)
        if handler is None:
            self._refuse(command, f"Unknown command {command!r}.")
            return None
        try:
            return handler(data or {})
        except Exception as error:  # noqa: BLE001 -- reported, never fatal
            logger.exception("[engine] %s raised", command)
            self._refuse(command, str(error))
            return None

    def _enqueue(self, data: dict) -> dict | None:
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            self._refuse("enqueue", "The prompt is empty; a clip needs one.")
            return None
        if self._generation.full:
            self._refuse(
                "enqueue",
                f"The generation queue is full ({self._generation.capacity} clips).",
            )
            return None

        seed = data.get("seed")
        if not isinstance(seed, int):
            # The stream's advancing default; an explicit seed leaves it
            # untouched, so explicit and automatic seeding do not interfere.
            seed = self._seed
            self._seed += 1
        seconds = data.get("seconds")
        frames = (clip_plan.frames_for_seconds(float(seconds)) if isinstance(seconds, int
                                                                             | float) else self._model.clip_frames)
        position = data.get("position")
        entry = new_entry(
            prompt=prompt,
            metadata=str(data.get("metadata") or ""),
            frames=frames,
            seed=seed,
        )
        self._generation.add(entry, position if isinstance(position, int) else None)
        self._emit("clip_queued", {"clip": entry.snapshot()})
        self._send_queue()
        self._send_state()
        return {"clip": entry.snapshot()}

    def _pop(self, data: dict) -> dict | None:
        """Take one clip out of whichever queue holds it."""
        clip_id = str(data.get("clip_id") or "")
        entry = ((self._generation.get(clip_id) or self._playout.get(clip_id)) if clip_id else None)
        if entry is None:
            self._refuse(
                "pop",
                f"No queued clip has id {clip_id!r}."
                if clip_id else "Pass the `clip_id` of the queued clip to remove.",
            )
            return None
        self._generation.remove(entry)
        self._playout.remove(entry)
        # A build already running for it finishes and is discarded; the queues
        # own what exists, so a result with no entry has nowhere to land.
        if self._build is not None and self._build[0] is entry:
            self._build[1].cancelled = True
        self._emit("clip_popped", {"clip": entry.snapshot()})
        self._send_queue()
        self._send_state()
        return {"clip": entry.snapshot()}

    def _move(self, data: dict) -> dict | None:
        clip_id = str(data.get("clip_id") or "")
        position = data.get("position")
        position = position if isinstance(position, int) else 0
        entry = self._generation.get(clip_id) if clip_id else None
        queue, name = (self._generation, "generation")
        if entry is None and clip_id:
            entry = self._playout.get(clip_id)
            queue, name = self._playout, "playout"
        if entry is None:
            self._refuse(
                "move",
                f"No queued clip has id {clip_id!r}." if clip_id else "Pass the `clip_id` of the queued clip to move.",
            )
            return None
        landed = queue.move(entry, position)
        self._send_queue()
        return {"clip": entry.snapshot(), "queue": name, "position": landed}

    # -------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        """Load the model, then generate and play forever.

        Loading is minutes of GPU work, so it runs on a thread: the web app is
        already serving by then, and a viewer sees the page rather than a
        connection refused.
        """
        height, width = self._canvas_hw()
        logger.info(
            "[engine] loading FastH3 on %d gpu(s) at %dx%d, %d-frame clips",
            int(self._model.runtime.get("num_gpus", 4)),
            width,
            height,
            self._model.clip_frames,
        )
        started = time.monotonic()
        await asyncio.to_thread(self.backend.load)
        logger.info("[engine] model ready in %.1fs", time.monotonic() - started)

        self._ready.set()
        self._send_state()
        self._send_queue()

        while True:
            try:
                self._pump_builds()
                entry = self._playout.head()
                if entry is not None:
                    self._playout.remove(entry)
                    self._send_queue()
                    await self._play_clip(entry)
                else:
                    await asyncio.sleep(POLL_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- the serve loop must survive anything
                logger.exception("[engine] error in the serve loop")
                await asyncio.sleep(POLL_SECONDS)

    # ------------------------------------------------------------- generation

    def _pump_builds(self) -> None:
        """Apply a finished build and keep the worker fed, without blocking.

        Called from the idle loop and from every playout slice, so clips keep
        building while another one streams. The generation queue is consumed
        front first, paused only while the playout queue is full -- a finished
        build needs a slot to land in, and that pause is the submit-time
        reservation which makes the later `add` impossible to overflow.
        """
        if self._build is not None:
            entry, job, submitted = self._build
            if not job.done.is_set():
                return
            self._build = None
            entry.building = False
            if job.cancelled or entry not in self._generation:
                # Its entry left the queue (a pop, or a preset flush) while the
                # build ran; the queues own what exists, so drop it.
                pass
            elif job.error is not None or job.result is None:
                # A finished, uncancelled job should always carry one or the
                # other. Reporting the empty case rather than unpacking it
                # keeps a future change to the worker from surfacing as a
                # TypeError inside the pump.
                reason = str(job.error) if job.error is not None else "the build produced no frames"
                self._generation.remove(entry)
                self._emit("clip_failed", {"clip": entry.snapshot(), "reason": reason})
                self._send_queue()
                self._send_state()
            else:
                entry.video, entry.audio = job.result
                self._generation.remove(entry)
                self._playout.add(entry)
                logger.info(
                    "[engine] clip generated: %s (%df) %.2fs after submit, "
                    "%d generating, %d playable",
                    entry.clip_id[:8],
                    entry.frames,
                    time.monotonic() - submitted,
                    len(self._generation),
                    len(self._playout),
                )
                self._emit("clip_generated", {"clip": entry.snapshot()})
                self._send_queue()
                self._send_state()

        if self._build is None and not self._playout.full:
            pending = self._generation.next_to_build()
            if pending is not None:
                height, width = self._canvas_hw()
                pending.building = True
                logger.info(
                    "[engine] clip build submitted: %s (%df), %d generating",
                    pending.clip_id[:8],
                    pending.frames,
                    len(self._generation),
                )
                self._build = (
                    pending,
                    self.backend.submit(
                        frames=pending.frames,
                        prompt=pending.prompt,
                        seed=pending.seed,
                        height=height,
                        width=width,
                    ),
                    time.monotonic(),
                )

    # ---------------------------------------------------------------- playout

    async def _play_clip(self, entry: ClipEntry) -> None:
        """Feed one built clip to the pacer at 24 fps, then report it done."""
        self._playing = entry
        try:
            self._emit("clip_started", {"clip": entry.snapshot()})
            self._send_state()
            await self._feed_clip(entry)
        finally:
            self._playing = None
            # The decoded frames are the bulk of this process's host memory;
            # dropping them here bounds it at the playout queue's capacity.
            entry.video, entry.audio = None, None
        self._clips_played += 1
        self._emit(
            "clip_finished",
            {
                "clip": entry.snapshot(),
                "seconds_sent": round(self._seconds_sent, 2)
            },
        )
        self._send_state()

    async def _feed_clip(self, entry: ClipEntry) -> None:
        """Hand the clip to the pacer in slices on a drift-free 24 fps clock.

        Paced by FRAMES rather than slices, because a clip's tail slice is
        short and charging it a whole slot would open a hole in the cadence.
        The clock is re-anchored rather than burst through: falling behind is
        a scheduling hiccup, and a catch-up burst would only overrun the
        pacer's buffers.

        Builds keep moving between slices, so the next clip is generating
        while this one plays.
        """
        import numpy as np

        pacer = self._pacer
        frames_list, samples = entry.video, entry.audio
        if pacer is None or not frames_list:
            return
        metadata = encode_id3(clip_view(entry.snapshot()))
        samples_per_frame = MODEL_SAMPLE_RATE / MODEL_FPS
        total = len(frames_list)
        clock_start: float | None = None
        frames_paced = 0
        loop = asyncio.get_running_loop()

        for lo in range(0, total, EMIT_FRAMES):
            self._pump_builds()
            hi = min(lo + EMIT_FRAMES, total)

            now = loop.time()
            if clock_start is None:
                clock_start = now
            content_pos = frames_paced / MODEL_FPS
            clock_start = max(clock_start, now - content_pos)
            delay = clock_start + content_pos - now
            if delay > 0:
                await asyncio.sleep(delay)

            for frame in frames_list[lo:hi]:
                pacer.submit_video(np.asarray(frame), metadata)
            if samples is not None:
                audio_lo = round(lo * samples_per_frame)
                audio_hi = round(hi * samples_per_frame)
                pacer.submit_audio(samples[:, audio_lo:audio_hi])

            frames_paced += hi - lo
            self._frames_sent += hi - lo
            self._seconds_sent = self._frames_sent / MODEL_FPS

        # Wait out the tail. The loop sleeps *before* each slice, so it exits
        # one slice-time early -- the last EMIT_FRAMES are pushed but never
        # paid for. That is a gain of EMIT_FRAMES/FPS on every clip, and since
        # the pacer drains at a flat 24 fps the surplus has nowhere to go but
        # its buffer: measured at ~0.16 s/min, which reaches the 2 s cap in
        # about twenty minutes and then starts dropping frames. Sleeping out
        # the remainder makes a clip cost exactly its own length, so feed rate
        # and drain rate are equal and the buffer depth is stationary.
        if clock_start is not None:
            tail = clock_start + total / MODEL_FPS - loop.time()
            if tail > 0:
                await asyncio.sleep(tail)


__all__ = ["MODEL_FPS", "MODEL_SAMPLE_RATE", "Engine"]
