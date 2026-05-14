# SPDX-License-Identifier: Apache-2.0
"""fMP4 stream encoder used by the streaming server.

The client's Media Source Extensions player needs a continuous fMP4
byte stream: first an *initialization segment* (``ftyp`` + ``moov``),
then one or more *media segments* (``moof`` + ``mdat``). We pipe raw
RGB frames into an ffmpeg subprocess configured for fragmented output
via ``-movflags empty_moov+default_base_moof+frag_keyframe+faststart``
and stream the bytes back out.
"""
from __future__ import annotations

import asyncio
import contextlib
import subprocess
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np


@dataclass
class FragmentedMP4Chunk:
    """A single fMP4 byte chunk emitted by :class:`FragmentedMP4Encoder`.

    ``kind`` identifies whether the chunk is the init segment (must be
    fed into the client's ``SourceBuffer`` first) or a media fragment.
    """

    kind: Literal["init", "media"]
    data: bytes
    stream_id: str
    segment_idx: int


class FragmentedMP4Encoder:
    """Stream RGB frames in, fMP4 chunks out.

    One encoder covers one segment. The server creates a new encoder
    per :class:`ltx2_segment_start`` boundary so each segment becomes
    one media fragment the client can append independently.

    Example::

        encoder = FragmentedMP4Encoder(width=1024, height=576, fps=24,
                                        segment_idx=0)
        async with encoder:
            async for chunk in encoder.encode(frames):
                await websocket.send_bytes(chunk.data)
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        segment_idx: int,
        stream_id: str | None = None,
        ffmpeg_path: str = "ffmpeg",
        preset: str = "ultrafast",
        pixel_format_out: str = "yuv420p",
        extra_args: list[str] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.segment_idx = segment_idx
        self.stream_id = stream_id or uuid.uuid4().hex
        self._ffmpeg_path = ffmpeg_path
        self._preset = preset
        self._pixel_format_out = pixel_format_out
        self._extra_args = list(extra_args or [])
        self._proc: subprocess.Popen | None = None
        self._init_emitted = False

    async def __aenter__(self) -> FragmentedMP4Encoder:
        self._spawn()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _spawn(self) -> None:
        args = [
            self._ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            self._preset,
            "-tune",
            "zerolatency",
            "-pix_fmt",
            self._pixel_format_out,
            "-movflags",
            "empty_moov+default_base_moof+frag_keyframe+faststart",
            "-f",
            "mp4",
            *self._extra_args,
            "-",
        ]
        # stderr → DEVNULL: with -loglevel error on, the only thing
        # stderr would carry is unsolicited warnings. Piping without a
        # reader deadlocks ffmpeg once the pipe buffer (~64 KiB) fills.
        self._proc = subprocess.Popen(  # noqa: S603
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    async def encode(
        self,
        frames: list[np.ndarray] | AsyncIterator[np.ndarray],
    ) -> AsyncIterator[FragmentedMP4Chunk]:
        """Feed frames into ffmpeg and yield fMP4 chunks as they appear."""
        if self._proc is None:
            self._spawn()
        assert self._proc is not None and self._proc.stdin is not None
        proc = self._proc

        loop = asyncio.get_running_loop()

        async def _writer() -> None:
            try:
                if hasattr(frames, "__aiter__"):
                    async for frame in frames:  # type: ignore[union-attr]
                        await loop.run_in_executor(None, _write_frame, proc.stdin, frame)
                else:
                    for frame in frames:  # type: ignore[assignment]
                        await loop.run_in_executor(None, _write_frame, proc.stdin, frame)
            finally:
                with contextlib.suppress(BrokenPipeError):
                    proc.stdin.close()

        writer_task = asyncio.create_task(_writer())
        try:
            reader = proc.stdout
            assert reader is not None
            # Read in reasonably-sized chunks; MSE tolerates any size
            # but we don't want to starve the event loop.
            chunk_size = 64 * 1024
            while True:
                data = await loop.run_in_executor(None, reader.read, chunk_size)
                if not data:
                    break
                kind: Literal["init", "media"] = "init" if not self._init_emitted else "media"
                self._init_emitted = True
                yield FragmentedMP4Chunk(
                    kind=kind,
                    data=bytes(data),
                    stream_id=self.stream_id,
                    segment_idx=self.segment_idx,
                )
        finally:
            await writer_task

    async def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except BrokenPipeError:
            pass
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, proc.wait),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await loop.run_in_executor(None, proc.wait)


def _write_frame(stdin, frame: np.ndarray) -> None:
    import numpy as np

    if not isinstance(frame, np.ndarray):
        raise TypeError("fMP4 encoder frames must be numpy.ndarray")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("fMP4 encoder frames must be HxWx3 uint8 RGB; got "
                         f"shape={frame.shape}, dtype={frame.dtype}")
    with contextlib.suppress(BrokenPipeError):
        stdin.write(frame.tobytes())


__all__ = [
    "FragmentedMP4Chunk",
    "FragmentedMP4Encoder",
]
