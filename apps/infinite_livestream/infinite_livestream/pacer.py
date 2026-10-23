"""The pacer: turn clip-shaped generation into a constant-rate broadcast.

Clips arrive in bursts and stop entirely between them; a live sink needs a
frame every period and audio every period, forever, or players stall. The
pacer is a drift-free metronome at the model's frame rate: each tick it pops
the oldest buffered frame (or repeats the last one, or black before anything
arrived) and pulls exactly one tick of samples (padding with silence).

Both buffers share one shallow cap, which is what keeps them together: while a
clip plays both sit near-empty and flow through with the same tiny delay;
while nothing plays both run dry. Depth here is end-to-end latency, so it is
deliberately small.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time

import numpy as np

from .metadata import EMPTY_ID3
from .sink import AudioFormat, HlsSink, VideoFormat

logger = logging.getLogger(__name__)

# How much media may sit between the model and the sink before the oldest is
# dropped. Shallow on purpose: depth here is end-to-end latency.
_BUFFER_SECONDS = 2.0

# If the loop is starved long enough to fall this many periods behind, resnap
# the clock instead of machine-gunning catch-up frames into the sink.
_RESNAP_PERIODS = 8


class Pacer:
    """Constant-rate A/V clock between the model callbacks and one sink."""

    def __init__(self, sink: HlsSink, video: VideoFormat, audio: AudioFormat) -> None:
        if audio.sample_rate % video.fps != 0:
            raise ValueError(f"sample rate {audio.sample_rate} must divide evenly by fps {video.fps}")
        self._sink = sink
        self._video = video
        self._audio = audio
        self._samples_per_tick = audio.sample_rate // video.fps

        max_frames = int(video.fps * _BUFFER_SECONDS)
        self._frames: collections.deque[tuple[np.ndarray, bytes]] = collections.deque(maxlen=max_frames)
        self._audio_chunks: collections.deque[np.ndarray] = collections.deque()
        self._audio_buffered = 0  # samples across _audio_chunks
        self._max_audio_samples = int(audio.sample_rate * _BUFFER_SECONDS)

        self._black = np.zeros((video.height, video.width, 3), dtype=np.uint8)
        self._silence = np.zeros(self._samples_per_tick, dtype=np.int16)
        self._last_frame = (self._black, EMPTY_ID3)

        # Counters, logged periodically and readable by anyone.
        self.ticks = 0
        self.repeated_frames = 0
        self.silent_ticks = 0
        self.dropped_frames = 0
        self.dropped_samples = 0
        # A video underflow run while audio still has data is the A/V sync
        # smell: the picture holds on a stale frame while the sound moves on.
        self._repeat_run = 0
        self._repeat_run_had_audio = 0
        self.worst_repeat_run = 0

    # ------------------------------------------------- model-facing intake

    def submit_video(self, frame: np.ndarray, metadata: bytes = EMPTY_ID3) -> None:
        """Buffer a frame together with its title, including through drops/repeats."""
        frame = np.asarray(frame)
        if frame.shape[:2] != (self._video.height, self._video.width):
            frame = self._fit(frame)
        if len(self._frames) == self._frames.maxlen:
            self.dropped_frames += 1
        self._frames.append((frame, metadata))

    def submit_audio(self, samples: np.ndarray) -> None:
        """Buffer model audio (int16, any chunk size; channels are flattened)."""
        flat = np.asarray(samples, dtype=np.int16).reshape(-1)
        if flat.size == 0:
            return
        self._audio_chunks.append(flat)
        self._audio_buffered += flat.size
        while self._audio_buffered > self._max_audio_samples:
            oldest = self._audio_chunks.popleft()
            self._audio_buffered -= oldest.size
            self.dropped_samples += oldest.size

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        """Center a differently-sized frame on the fixed black canvas.

        Raw-video geometry cannot change mid-stream, so an odd-sized frame is
        letterboxed rather than resized: no interpolation dependency, and it
        cannot garble the stream.
        """
        height, width = self._video.height, self._video.width
        crop = frame[:height, :width, :3]
        canvas = self._black.copy()
        top = (height - crop.shape[0]) // 2
        left = (width - crop.shape[1]) // 2
        canvas[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
        return canvas

    def _pull_audio_tick(self) -> np.ndarray:
        """Exactly one tick of samples: buffered audio padded with silence."""
        needed = self._samples_per_tick
        if self._audio_buffered == 0:
            self.silent_ticks += 1
            return self._silence
        parts: list[np.ndarray] = []
        while needed > 0 and self._audio_chunks:
            chunk = self._audio_chunks[0]
            if chunk.size <= needed:
                parts.append(self._audio_chunks.popleft())
                needed -= chunk.size
            else:
                parts.append(chunk[:needed])
                self._audio_chunks[0] = chunk[needed:]
                needed = 0
        pulled = np.concatenate(parts) if len(parts) > 1 else parts[0]
        self._audio_buffered -= pulled.size
        if needed > 0:
            pulled = np.concatenate([pulled, np.zeros(needed, dtype=np.int16)])
        return pulled

    # ------------------------------------------------------------ the clock

    async def run(self) -> None:
        """Tick forever at the frame rate; cancelled only at shutdown."""
        await self._sink.start(self._video, self._audio)
        period = 1.0 / self._video.fps
        next_tick = time.monotonic() + period
        last_report = time.monotonic()

        while True:
            delay = next_tick - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            elif -delay > period * _RESNAP_PERIODS:
                logger.warning("[pacer] %.2fs behind schedule; resnapping the clock", -delay)
                next_tick = time.monotonic()
            next_tick += period

            if self._frames:
                if self._repeat_run:
                    if self._repeat_run_had_audio >= 2:
                        logger.info(
                            "[pacer] picture held %.2fs while %.2fs of audio played on",
                            self._repeat_run / self._video.fps,
                            self._repeat_run_had_audio / self._video.fps,
                        )
                    self.worst_repeat_run = max(self.worst_repeat_run, self._repeat_run)
                    self._repeat_run = 0
                    self._repeat_run_had_audio = 0
                self._last_frame = self._frames.popleft()
            else:
                self.repeated_frames += 1
                self._repeat_run += 1
                if self._audio_buffered > 0:
                    self._repeat_run_had_audio += 1
            self._sink.send_video(*self._last_frame)
            self._sink.send_audio(self._pull_audio_tick())
            self.ticks += 1

            now = time.monotonic()
            if now - last_report >= 60.0:
                # Buffer depths are the A/V sync diagnostic: the two are only
                # in sync while both sit near zero. A standing audio depth with
                # an empty video buffer means audio is playing that many
                # seconds ahead of the picture it belongs to.
                logger.info(
                    "[pacer] buffers: video %.2fs (%d frames) audio %.2fs (%d samples)",
                    len(self._frames) / self._video.fps,
                    len(self._frames),
                    self._audio_buffered / self._audio.sample_rate,
                    self._audio_buffered,
                )
                logger.info(
                    "[pacer] ticks=%d live_frames=%d repeats=%d "
                    "silent_ticks=%d dropped=%df/%.1fs-audio",
                    self.ticks,
                    self.ticks - self.repeated_frames,
                    self.repeated_frames,
                    self.silent_ticks,
                    self.dropped_frames,
                    self.dropped_samples / self._audio.sample_rate,
                )
                last_report = now
