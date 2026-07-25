"""Video-only fragmented-MP4 encoding and completed-prefix finalization."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from collections.abc import Iterable
import uuid

import numpy as np

MEDIA_MIME = 'video/mp4; codecs="avc1.42E01E"'


@dataclass(frozen=True, slots=True)
class EncodedMediaSegment:
    block_index: int
    init_bytes: bytes
    media_bytes: bytes
    path: Path
    mime: str = MEDIA_MIME


def split_fmp4(data: bytes) -> tuple[bytes, bytes]:
    """Split top-level fMP4 initialization boxes from moof/mdat media."""
    offset = 0
    first_fragment: int | None = None
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset:offset + 4], "big")
        box_type = data[offset + 4:offset + 8]
        header = 8
        if size == 1:
            if offset + 16 > len(data):
                break
            size = int.from_bytes(data[offset + 8:offset + 16], "big")
            header = 16
        elif size == 0:
            size = len(data) - offset
        if size < header or offset + size > len(data):
            raise ValueError("invalid top-level MP4 box")
        if box_type == b"moof":
            first_fragment = offset
            break
        offset += size
    if first_fragment is None:
        raise ValueError("ffmpeg output did not contain an fMP4 moof box")
    init_bytes = data[:first_fragment]
    media_bytes = data[first_fragment:]
    if b"ftyp" not in init_bytes or b"moov" not in init_bytes:
        raise ValueError("ffmpeg output is missing fMP4 initialization boxes")
    return init_bytes, media_bytes


class FMP4BlockWriter:

    def __init__(
        self,
        output_root: str | os.PathLike[str],
        *,
        fps: float,
        ffmpeg_bin: str | None = None,
    ) -> None:
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        resolved_ffmpeg = ffmpeg_bin or shutil.which(os.getenv("WANTRACK_FFMPEG_BIN", "ffmpeg"))
        if not resolved_ffmpeg:
            raise RuntimeError("ffmpeg is required for WanTrack streaming")
        self.ffmpeg_bin = resolved_ffmpeg
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self.session_dir = root / self.session_id
        self.session_dir.mkdir()
        self._block_paths: list[Path] = []

    @property
    def block_paths(self) -> tuple[Path, ...]:
        return tuple(self._block_paths)

    def encode_block(
        self,
        frames: np.ndarray,
        block_index: int,
    ) -> EncodedMediaSegment:
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape [T, H, W, 3]")
        if frames.shape[0] <= 0:
            raise ValueError("cannot encode an empty frame block")
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        height, width = int(frames.shape[1]), int(frames.shape[2])
        gop = max(1, int(frames.shape[0]))
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-movflags",
            "+empty_moov+default_base_moof+frag_keyframe",
            "-frag_duration",
            str(max(1, round(1_000_000 * frames.shape[0] / self.fps))),
            "-f",
            "mp4",
            "pipe:1",
        ]
        result = subprocess.run(
            command,
            input=frames.tobytes(),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to encode WanTrack block {block_index}: "
                               f"{stderr or f'exit {result.returncode}'}")
        init_bytes, media_bytes = split_fmp4(result.stdout)
        path = self.session_dir / f"block_{int(block_index):06d}.mp4"
        path.write_bytes(result.stdout)
        self._block_paths.append(path)
        return EncodedMediaSegment(
            block_index=int(block_index),
            init_bytes=init_bytes,
            media_bytes=media_bytes,
            path=path,
        )

    def finalize(self) -> Path | None:
        if not self._block_paths:
            return None
        output_path = self.session_dir / "wantrack_control.mp4"
        concat_path = self.session_dir / "concat.txt"
        concat_path.write_text(
            "".join(f"file '{self._concat_escape(path)}'\n" for path in self._block_paths),
            encoding="utf-8",
        )
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-an",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not output_path.is_file():
            return self._finalize_reencode(output_path)
        return output_path if output_path.stat().st_size > 0 else None

    @staticmethod
    def _concat_escape(path: Path) -> str:
        return str(path.resolve()).replace("'", "'\\''")

    def _finalize_reencode(self, output_path: Path) -> Path | None:
        concat_path = self.session_dir / "concat.txt"
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
        )
        if (result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0):
            return output_path
        return None


def total_size(paths: Iterable[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())
