"""Encode the paced stream with ffmpeg and write it as an HLS playlist.

The page serves the playlist itself, so one HTTP origin (and one tunnel)
carries the whole demo. Latency is a segment plus the player's buffer, which
is irrelevant here because clips are pre-built anyway.

The pacer calls `send_video` once per frame period with one rgb24 frame of the
fixed size and `send_audio` once per period with one period of int16 samples,
forever. Four things about that contract are load-bearing:

  * ffmpeg reads both pipes as raw untimestamped bytes and derives every PTS
    from the byte count, so an entry dropped on one pipe and not the other
    shifts sound against picture permanently. Both are gated on the same
    `_ensure_running`, and any residual imbalance is logged as the skew it
    will cost. It cannot be repaired afterwards by withholding from the other
    pipe -- that starves ffmpeg's muxer and stalls the stream.
  * A frame whose byte count disagrees with `-s WxH` shifts every following
    scanline and the picture turns to static, so wrong-sized frames are
    refused rather than written.
  * `stdin.write` blocks when ffmpeg's input buffer fills, and blocking the
    event loop snowballs. Each pipe gets a writer thread behind a bounded
    queue.
  * ffmpeg exits on transient errors; the stream must not. It is restarted
    lazily on the next frame, with a cooldown and a failure cap.

Requires ffmpeg on PATH. Uses `pass_fds`, so Linux/macOS only.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np

from .metadata import EMPTY_ID3
from .muxer import SEGMENT_SECONDS, MetadataMuxer

logger = logging.getLogger("infinite_livestream.sink")

_RESTART_COOLDOWN_S = 2.0
_MAX_CONSECUTIVE_FAILURES = 5
_PROCESS_EXIT_TIMEOUT_S = 2.0
_WRITER_EXIT_TIMEOUT_S = 2.0

# Writer-queue depth. Not latency -- the pacer governs the rate -- only
# headroom for the seconds x264 spends starting up. Video gets more because a
# video entry is a whole raw frame (3.1 MB at 1344x768) against 4 KB of audio,
# so it is the only queue that ever overflows.
_QUEUE_SECONDS = 2.0
_VIDEO_QUEUE_SECONDS = 8.0

# Let ffmpeg open its encoder before the first frame. Without it the pacer
# pushes 24 fps of raw frames into a process that is not reading yet, and the
# queue oversubscribes before a single frame is consumed.
_ENCODER_SETTLE_S = 2.0


@dataclass(frozen=True)
class VideoFormat:
    """Geometry and rate of the paced video stream."""

    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class AudioFormat:
    """Sample layout of the paced audio stream (int16 PCM)."""

    sample_rate: int
    channels: int


class _PipeWriter(threading.Thread):
    """Feed one ffmpeg input pipe from a bounded queue, off the event loop."""

    def __init__(self, name: str, maxsize: int) -> None:
        super().__init__(name=f"sink-{name}", daemon=True)
        self.queue: queue.Queue[tuple[bytes, bytes | None] | None] = queue.Queue(maxsize=maxsize)
        self.pipe: IO[bytes] | None = None
        self.metadata_queue: queue.Queue[bytes] | None = None
        self.broken = threading.Event()
        self.dropped = 0
        self._lock = threading.Lock()

    def attach(self, pipe, metadata_queue: queue.Queue[bytes] | None = None) -> None:
        with self._lock:
            self.pipe = pipe
            self.metadata_queue = metadata_queue
            self.broken.clear()

    def submit(self, payload: bytes, metadata: bytes | None = None) -> int:
        """Enqueue bytes, dropping the oldest rather than ever blocking.

        Returns how many entries were shed for A/V skew diagnostics. Metadata
        stays with its payload; discarded frames never enter the muxer ledger.
        """
        shed = 0
        try:
            self.queue.put_nowait((payload, metadata))
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped += 1
                shed += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((payload, metadata))
            except queue.Full:
                self.dropped += 1
                shed += 1
        return shed

    def flush(self) -> None:
        """Discard everything queued, so a restart resumes both pipes level."""
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:  # shutdown sentinel
                return
            payload, metadata = item
            with self._lock:
                pipe = self.pipe
                metadata_queue = self.metadata_queue
            if pipe is None or self.broken.is_set():
                continue  # ffmpeg is down; discard until it is restarted
            try:
                # Unbuffered pipes can return a short write. Every byte must
                # reach ffmpeg or its raw frame/sample boundaries shift.
                remaining = memoryview(payload)
                while remaining:
                    written = pipe.write(remaining)
                    if not written:
                        raise BrokenPipeError("ffmpeg input pipe stopped accepting data")
                    remaining = remaining[written:]
                if metadata_queue is not None and metadata is not None:
                    metadata_queue.put_nowait(metadata)
            except (BrokenPipeError, OSError, ValueError, queue.Full):
                # ValueError: write to a closed file during a restart race.
                with self._lock:
                    if self.pipe is pipe:
                        self.broken.set()

    def close(self) -> None:
        # No producer runs during shutdown. Discard queued media so the
        # sentinel never waits for a writer blocked inside pipe.write().
        self.flush()
        self.queue.put_nowait(None)


class HlsSink:
    """Write the paced stream as an HLS playlist under `directory`."""

    def __init__(self,
                 directory: str | Path,
                 video_bitrate_k: int = 4500,
                 *,
                 playlist_name: str = "stream.m3u8",
                 retention_s: int = 120) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; install it first")
        self._directory = Path(directory)
        self._playlist_name = playlist_name
        self._bitrate_k = video_bitrate_k
        if retention_s < SEGMENT_SECONDS * 3:
            raise ValueError("HLS retention must cover at least three segments")
        self._retention_s = retention_s
        self._muxer: MetadataMuxer | None = None
        self._video: VideoFormat | None = None
        self._audio: AudioFormat | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._audio_pipe: IO[bytes] | None = None
        self._video_writer: _PipeWriter | None = None
        self._audio_writer: _PipeWriter | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        self._failures = 0
        self._last_start_attempt = 0.0
        self._frames_sent = 0
        self._dead = False
        self._video_shed = 0
        self._audio_shed = 0

    @property
    def playlist_path(self) -> Path:
        """Where the web app points the player."""
        return self._directory / self._playlist_name

    # ------------------------------------------------------------ lifecycle

    async def start(self, video: VideoFormat, audio: AudioFormat) -> None:
        self._video = video
        self._audio = audio
        self._video_writer = _PipeWriter("video", maxsize=int(video.fps * _VIDEO_QUEUE_SECONDS))
        self._audio_writer = _PipeWriter("audio", maxsize=int(video.fps * _QUEUE_SECONDS))
        self._video_writer.start()
        self._audio_writer.start()
        self._spawn_ffmpeg()
        # Awaited before the pacer's first tick, so this costs nothing.
        await asyncio.sleep(_ENCODER_SETTLE_S)

    def _spawn_ffmpeg(self) -> None:
        assert self._video is not None and self._audio is not None
        video, audio = self._video, self._audio
        self._last_start_attempt = time.monotonic()

        self._directory.mkdir(parents=True, exist_ok=True)

        audio_read_fd, audio_write_fd = os.pipe()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            # video in: raw rgb24 on stdin
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{video.width}x{video.height}",
            "-r",
            str(video.fps),
            "-i",
            "pipe:0",
            # audio in: raw int16 PCM on an inherited pipe
            "-f",
            "s16le",
            "-ar",
            str(audio.sample_rate),
            "-ac",
            str(audio.channels),
            "-i",
            f"pipe:{audio_read_fd}",
            "-map",
            "0:v",
            "-map",
            "1:a",
            # Preserve input frame order/count for the metadata ledger.
            "-fps_mode",
            "passthrough",
            "-bf",
            "0",
            "-sc_threshold",
            "0",
            # video encode
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",  # players cannot take 4:4:4
            "-g",
            str(video.fps * SEGMENT_SECONDS),  # a keyframe per segment
            "-b:v",
            f"{self._bitrate_k}k",
            "-maxrate",
            f"{int(self._bitrate_k * 1.2)}k",
            "-bufsize",
            f"{self._bitrate_k * 2}k",
            # audio encode
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "44100",
            "-ac",
            "2",
            # A persistent muxer adds timed metadata without re-encoding.
            "-f",
            "mpegts",
            "-muxdelay",
            "0",
            "-flush_packets",
            "1",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(cmd,
                                             stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             bufsize=0,
                                             pass_fds=(audio_read_fd, ))
        except Exception:
            os.close(audio_write_fd)
            raise
        finally:
            os.close(audio_read_fd)  # the child inherited its own copy

        audio_pipe = os.fdopen(audio_write_fd, "wb", buffering=0)
        self._audio_pipe = audio_pipe
        assert self._video_writer and self._audio_writer
        # Whatever each queue still held belonged to the dead ffmpeg, and the
        # two held different amounts; carrying it over starts the new one out
        # of sync.
        self._video_writer.flush()
        self._audio_writer.flush()
        self._video_shed = self._audio_shed = 0
        assert self._process.stdout is not None
        self._muxer = MetadataMuxer(self._process.stdout, self.playlist_path, video.fps, self._retention_s)
        self._video_writer.attach(self._process.stdin, self._muxer.frames)
        self._audio_writer.attach(audio_pipe)
        self._muxer.start()

        threading.Thread(target=self._drain_stderr, args=(self._process, ), daemon=True, name="sink-stderr").start()
        logger.info("[sink] ffmpeg started: %dx%d@%dfps -> %s", video.width, video.height, video.fps,
                    self.playlist_path)

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stderr is not None
        with process.stderr:
            for raw in process.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    self._stderr_tail.append(line)

    # ----------------------------------------------------------- restarting

    def _ensure_running(self) -> bool:
        """True when ffmpeg is up; otherwise try to restart it (rate-limited)."""
        if self._dead:
            return False
        process = self._process
        writers_broken = bool((self._video_writer and self._video_writer.broken.is_set())
                              or (self._audio_writer and self._audio_writer.broken.is_set()))
        muxer_finished = self._muxer is not None and self._muxer.finished.is_set()
        if process is not None and process.poll() is None and not writers_broken and not muxer_finished:
            return True

        if process is not None and (process.poll() is not None or writers_broken or muxer_finished):
            tail = "\n".join(list(self._stderr_tail)[-8:])
            logger.warning("[sink] ffmpeg died (exit=%s)%s", process.poll(), f"\n{tail}" if tail else "")
            self._teardown_process()

        if time.monotonic() - self._last_start_attempt < _RESTART_COOLDOWN_S:
            return False
        try:
            self._spawn_ffmpeg()
            self._failures = 0
            return True
        except Exception as error:
            self._failures += 1
            logger.error("[sink] restart failed (%d/%d): %s", self._failures, _MAX_CONSECUTIVE_FAILURES, error)
            if self._failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.error("[sink] giving up; the stream is dead")
                self._dead = True
            return False

    def _teardown_process(self) -> None:
        process, self._process = self._process, None
        muxer, self._muxer = self._muxer, None
        if muxer is not None:
            muxer.cancelled.set()
        audio_pipe, self._audio_pipe = self._audio_pipe, None
        if process is None:
            return
        # Stop the reader before closing its inputs: a buffered close used
        # to wait behind a blocked write while ffmpeg was still alive.
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        # Close the unbuffered inputs now so an encoder waiting for data can
        # observe EOF and finish. These wrappers have no buffered-write lock;
        # keeping them also prevents a delayed write from reusing a closed fd.
        for pipe in (process.stdin, audio_pipe):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()
        try:
            process.wait(timeout=_PROCESS_EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            try:
                process.wait(timeout=_PROCESS_EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                logger.error("[sink] ffmpeg did not exit after SIGKILL")

        if muxer is not None:
            muxer.join(timeout=_WRITER_EXIT_TIMEOUT_S)
            if muxer.is_alive():
                self._dead = True
                raise RuntimeError("metadata muxer did not stop; refusing concurrent playlist writers")
        if process.stdout is not None:
            process.stdout.close()

    # ------------------------------------------------------------- delivery

    def send_video(self, frame: np.ndarray, metadata: bytes = EMPTY_ID3) -> None:
        if not self._ensure_running():
            return
        video = self._video
        assert video is not None and self._video_writer is not None
        if frame.shape[0] != video.height or frame.shape[1] != video.width:
            logger.error("[sink] refusing %sx%s frame (expected %dx%d)", frame.shape[1], frame.shape[0], video.width,
                         video.height)
            return
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        self._video_shed += self._video_writer.submit(frame.tobytes(), metadata)
        self._frames_sent += 1
        if self._frames_sent % (video.fps * 60) == 0:
            logger.info(
                "[sink] %d frames sent (dropped: %d video / %d audio; net A/V skew %+.3fs; "
                "queue %d)",
                self._frames_sent,
                self._video_writer.dropped,
                self._audio_writer.dropped if self._audio_writer else 0,
                # What ffmpeg's byte-counted PTS is out by. Zero is the point.
                (self._video_shed - self._audio_shed) / video.fps,
                self._video_writer.queue.qsize(),
            )

    def send_audio(self, samples: np.ndarray) -> None:
        # Gated exactly like send_video: audio written while video is withheld
        # would run ahead by that outage once ffmpeg came back.
        if self._audio_writer is None or not self._ensure_running():
            return
        self._audio_shed += self._audio_writer.submit(np.ascontiguousarray(samples, dtype=np.int16).tobytes())

    async def stop(self) -> None:
        self._dead = True
        for writer in (self._video_writer, self._audio_writer):
            if writer:
                writer.close()
        await asyncio.to_thread(self._teardown_process)
        for writer in (self._video_writer, self._audio_writer):
            if writer is not None and writer.ident is not None:
                await asyncio.to_thread(writer.join, _WRITER_EXIT_TIMEOUT_S)
                if writer.is_alive():
                    logger.warning("[sink] %s did not stop within the shutdown timeout", writer.name)
        logger.info("[sink] stopped after %d frames", self._frames_sent)
