"""Copy encoded packets to HLS, attaching titles to the video packet's own PTS."""

from __future__ import annotations

import contextlib
import logging
import math
import queue
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import IO

import av

logger = logging.getLogger(__name__)
SEGMENT_SECONDS = 2


class MetadataMuxer(threading.Thread):
    """One encoder generation, one ordered ledger, and one persistent HLS muxer.

    FFmpeg preserves input frame order and count. The writer commits one ID3
    record after each complete raw frame write; this thread pairs those records
    with encoded video packets. No enqueue times or wall-clock offsets are used.
    """

    def __init__(self, source: IO[bytes], playlist: Path, fps: int, retention_s: int) -> None:
        super().__init__(name="sink-muxer", daemon=True)
        self.source = source
        self.playlist = playlist
        self.fps = fps
        self.retention_s = retention_s
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=fps * 30)
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.error: Exception | None = None
        self.epoch = uuid.uuid4().hex
        self._last_cleanup = 0.0

    def _next_metadata(self) -> bytes | None:
        deadline = time.monotonic() + 5.0
        while not self.cancelled.is_set():
            try:
                return self.frames.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise RuntimeError("encoded frame has no committed title record") from None
        return None

    def _cleanup(self) -> None:
        """Reap orphaned files from old encoders; never delete listed segments."""
        now = time.monotonic()
        if now - self._last_cleanup < SEGMENT_SECONDS:
            return
        self._last_cleanup = now
        try:
            listed = {line.strip() for line in self.playlist.read_text().splitlines() if not line.startswith("#")}
        except OSError:
            return
        cutoff = time.time() - self.retention_s
        for path in self.playlist.parent.glob("seg_*.ts*"):
            if path.name in listed or self.epoch in path.name:
                continue
            with contextlib.suppress(OSError):
                if path.stat().st_mtime < cutoff:
                    path.unlink()

    def run(self) -> None:
        try:
            self._mux()
            if not self.cancelled.is_set():
                raise RuntimeError("encoder output ended")
        except Exception as error:
            if not self.cancelled.is_set():
                self.error = error
                logger.exception("[sink] metadata muxer failed")
        finally:
            self.finished.set()

    def _mux(self) -> None:
        options = {
            "hls_time": str(SEGMENT_SECONDS),
            "hls_list_size": str(max(3, math.ceil(self.retention_s / SEGMENT_SECONDS))),
            "hls_segment_filename": str(self.playlist.parent / f"seg_{self.epoch}_%010d.ts"),
            "hls_segment_options": "mpegts_copyts=1",
            "hls_flags": "delete_segments+independent_segments+omit_endlist+temp_file+append_list",
            "avoid_negative_ts": "disabled",
            # Sparse ID3 must not hold seconds of video in the interleaver.
            "max_interleave_delta": "100000",
        }
        # Limit format probing: the input is a known MPEG-TS stream with H.264
        # and AAC, not a file whose format needs seconds of discovery.
        with av.open(self.source, format="mpegts", options={
                "probesize": "65536",
                "analyzeduration": "1000000"
        }) as source, av.open(str(self.playlist), "w", format="hls", options=options) as output:
            streams = {
                s.index: output.add_stream_from_template(s)
                for s in source.streams if s.type in ("video", "audio")
            }
            metadata = output.add_data_stream("timed_id3")
            metadata.time_base = Fraction(1, 90000)
            last_record = None
            first_pts = None
            frame_number = 0
            for packet in source.demux():
                if self.cancelled.is_set():
                    break
                if packet.dts is None or packet.stream.index not in streams:
                    continue
                if packet.stream.type == "video":
                    pts = packet.pts * packet.time_base
                    if first_pts is None:
                        first_pts = pts
                    # This checks the encoder's one-frame-in/one-frame-out
                    # contract before metadata can be attached incorrectly.
                    expected = first_pts + Fraction(frame_number, self.fps)
                    if abs(pts - expected) > Fraction(1, 90000):
                        raise RuntimeError("encoder changed video frame cadence")
                    record = self._next_metadata()
                    if record is None:
                        break
                    # Every keyframe carries a complete record so every HLS
                    # segment is independently joinable, even mid-clip.
                    if packet.is_keyframe or record != last_record:
                        tag = av.Packet(record)
                        tag.stream = metadata
                        tag.time_base = packet.time_base
                        tag.pts = packet.pts
                        tag.dts = packet.dts
                        output.mux(tag)
                        last_record = record
                    frame_number += 1
                packet.stream = streams[packet.stream.index]
                output.mux(packet)
                self._cleanup()
