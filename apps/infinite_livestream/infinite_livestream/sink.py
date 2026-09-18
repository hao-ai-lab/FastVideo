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
from datetime import datetime
from pathlib import Path
from typing import IO

import numpy as np

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

# One segment is the floor on how fresh a viewer's picture can be. Six of them
# gives ~12 s of rewind and bounds the directory, since `delete_segments`
# removes them from disk as they roll off.
SEGMENT_SECONDS = 2
PLAYLIST_SEGMENTS = 6


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
        self.queue: queue.Queue[bytes | None] = queue.Queue(maxsize=maxsize)
        self.pipe: IO[bytes] | None = None
        self.broken = threading.Event()
        self.dropped = 0
        self._lock = threading.Lock()

    def attach(self, pipe) -> None:
        with self._lock:
            self.pipe = pipe
            self.broken.clear()

    def submit(self, payload: bytes) -> int:
        """Enqueue bytes, dropping the oldest rather than ever blocking.

        Returns how many entries were shed; the caller must shed the same
        number from the other stream or the two drift apart for good.
        """
        shed = 0
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped += 1
                shed += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(payload)
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
            payload = self.queue.get()
            if payload is None:  # shutdown sentinel
                return
            with self._lock:
                pipe = self.pipe
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
            except (BrokenPipeError, OSError, ValueError):
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
                 playlist_name: str = "stream.m3u8") -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; install it first")
        self._directory = Path(directory)
        self._playlist_name = playlist_name
        self._bitrate_k = video_bitrate_k
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
        # Wall clock ffmpeg gave output frame 0, read back from the playlist it
        # writes, and the frames submitted since. Together they give the
        # PROGRAM-DATE-TIME a frame entering now will carry -- see
        # `stream_time`. Not predictable: measured at 8.8 s after the process
        # starts and 6.8 s after its first frame, so it has to be read.
        self._pdt_base: float | None = None
        self._pdt_base_checked = 0.0
        self._stream_frames = 0
        self._dead = False
        self._video_shed = 0
        self._audio_shed = 0

    @property
    def playlist_path(self) -> Path:
        """Where the web app points the player."""
        return self._directory / self._playlist_name

    def stream_time(self) -> float | None:
        """The PROGRAM-DATE-TIME a frame handed over now will carry.

        The web app stamps its timeline with this, and the page compares its
        own playback position against those stamps, so the two have to mean the
        same instant. (Most browsers cannot report a date at all, and the page
        falls back to `published_until` for those -- but the timeline is keyed
        this way either way.)

        Frame counting is what makes them agree: ffmpeg's output frame 0 is the
        first frame submitted here, so a frame submitted now lands at
        `base + n / fps`. The base is *read* from ffmpeg rather than computed,
        because the offset between starting the process and the date it stamps
        is not something the caller can know -- measured at 8.8 s here, against
        a 2 s encoder settle.

        None until the first segment is published, when there is no PDT to be
        positioned against anyway.
        """
        if self._video is None:
            return None
        if self._pdt_base is None:
            self._learn_pdt_base()
            if self._pdt_base is None:
                return None
        return self._pdt_base + self._stream_frames / self._video.fps

    def _learn_pdt_base(self) -> None:
        """Read the wall clock ffmpeg gave output frame 0, from its playlist.

        Segment `n` starts one nominal segment-length after segment `n-1`, so
        the base is the first date in the playlist wound back by the media
        sequence. Read once per ffmpeg and as early as possible: segments that
        have already rolled off can only be accounted for at their nominal
        length, and a live one occasionally runs long.
        """
        now = time.monotonic()
        if now - self._pdt_base_checked < 1.0:
            return
        self._pdt_base_checked = now
        try:
            text = self.playlist_path.read_text()
        except OSError:
            return
        sequence, first = 0, None
        for line in text.splitlines():
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                sequence = int(line.split(":", 1)[1])
            elif line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                first = datetime.fromisoformat(line.split(":", 1)[1].strip()).timestamp()
                break
        if first is None:
            return
        self._pdt_base = first - sequence * SEGMENT_SECONDS
        logger.info("[sink] stream clock anchored at %.3f (from segment %d)", self._pdt_base, sequence)

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

        # A restart must not leave a player reading a playlist that references
        # segments from before the gap.
        shutil.rmtree(self._directory, ignore_errors=True)
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
            # output: a sliding-window playlist
            "-f",
            "hls",
            "-hls_time",
            str(SEGMENT_SECONDS),
            "-hls_list_size",
            str(PLAYLIST_SEGMENTS),
            # delete_segments bounds the directory, independent_segments lets a
            # player start anywhere, omit_endlist keeps it live rather than
            # signalling a finished VOD, program_date_time is what the page
            # uses to line the picture up with the now-playing title.
            "-hls_flags",
            "delete_segments+independent_segments+omit_endlist+program_date_time",
            "-hls_segment_type",
            "mpegts",
            "-hls_segment_filename",
            str(self._directory / "seg_%05d.ts"),
            str(self.playlist_path),
        ]
        try:
            self._process = subprocess.Popen(cmd,
                                             stdin=subprocess.PIPE,
                                             stdout=subprocess.DEVNULL,
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
        self._video_writer.attach(self._process.stdin)
        self._audio_writer.attach(audio_pipe)

        threading.Thread(target=self._drain_stderr, args=(self._process, ), daemon=True, name="sink-stderr").start()
        # The clock re-anchors from the new playlist, which this spawn wiped.
        self._pdt_base = None
        self._pdt_base_checked = 0.0
        self._stream_frames = 0
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
        if process is not None and process.poll() is None and not writers_broken:
            return True

        if process is not None and (process.poll() is not None or writers_broken):
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

    # ------------------------------------------------------------- delivery

    def send_video(self, frame: np.ndarray) -> None:
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
        if self._pdt_base is None:
            # Anchor as early as the first segment allows: the base is wound
            # back from the newest date by the media sequence at a nominal
            # segment length, and a live segment occasionally runs long, so a
            # long wind-back accumulates error. Self-throttled, and this stops
            # touching the disk entirely once anchored.
            self._learn_pdt_base()
        self._video_shed += self._video_writer.submit(frame.tobytes())
        self._frames_sent += 1
        self._stream_frames += 1
        if self._frames_sent % (video.fps * 60) == 0:
            logger.info(
                "[sink] %d frames sent (dropped: %d video / %d audio; net A/V skew %+.3fs; "
                "queue %d; clock %+.2fs vs playlist)",
                self._frames_sent,
                self._video_writer.dropped,
                self._audio_writer.dropped if self._audio_writer else 0,
                # What ffmpeg's byte-counted PTS is out by. Zero is the point.
                (self._video_shed - self._audio_shed) / video.fps,
                self._video_writer.queue.qsize(),
                self._clock_drift(),
            )

    def published_until(self) -> float | None:
        """The date of the newest frame a player can actually have.

        A viewer's position cannot be read back from every browser -- the
        PROGRAM-DATE-TIME APIs are inconsistent and some expose nothing -- so
        the page needs a second way to locate itself: this, minus how far the
        viewer is behind their own buffer edge, puts them on the same clock
        without asking the player for a date at all.
        """
        try:
            newest, span = 0.0, 0.0
            for line in self.playlist_path.read_text().splitlines():
                if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                    newest = datetime.fromisoformat(line.split(":", 1)[1].strip()).timestamp()
                elif line.startswith("#EXTINF:"):
                    span = float(line.split(":", 1)[1].rstrip(","))
            return (newest + span) if newest else None
        except OSError:
            return None

    def _clock_drift(self) -> float:
        """How far `stream_time` sits ahead of the published video, for the log."""
        published, now = self.published_until(), self.stream_time()
        return (now - published) if (now and published) else 0.0

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
