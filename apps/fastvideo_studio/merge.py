# SPDX-License-Identifier: Apache-2.0
"""Join a scene's clips, in order, into one video.

Kept apart from ``server.py`` because it needs nothing from FastVideo -- only
PyAV (to read each clip's format) and an ``ffmpeg`` binary -- so it can be
exercised without a GPU.

Clips from one scene share codec, size and frame rate, so they are joined by
copying the streams rather than re-encoding: it takes about a second, loses no
quality, and keeps picture and sound in step. Each clip is cut off at the end of
its last video frame, because the audio track runs a few milliseconds past it and
would otherwise open a gap between clips.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
MERGED_DIRNAME = "merged"
_FILENAME_RE = re.compile(r"^[\w.-]+\.mp4$")


class MergeError(Exception):
    """A merge that can't be done, carrying the HTTP status to report."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def find_ffmpeg() -> str | None:
    """An ``ffmpeg`` to run: $FASTVIDEO_FFMPEG_BIN, then PATH, then imageio-ffmpeg's bundled one."""
    configured = os.getenv("FASTVIDEO_FFMPEG_BIN")
    if configured and (shutil.which(configured) or os.path.isfile(configured)):
        return shutil.which(configured) or configured
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


@dataclass(frozen=True)
class ClipInfo:
    path: str
    video_codec: str
    width: int
    height: int
    fps: float
    pix_fmt: str
    video_seconds: float
    audio: tuple[str, int, int] | None  # codec, sample rate, channels

    def describe(self) -> str:
        audio = f"{self.audio[0]} {self.audio[1]} Hz x{self.audio[2]}" if self.audio else "no audio"
        return f"{self.width}x{self.height} {self.fps:.2f} fps {self.video_codec}/{self.pix_fmt}, {audio}"

    def signature(self) -> tuple[Any, ...]:
        return (self.video_codec, self.width, self.height, round(self.fps, 3), self.pix_fmt, self.audio)


def probe_clip(path: str) -> ClipInfo:
    """Read what ``check_compatible`` needs from a video file."""
    import av

    try:
        with av.open(path) as container:
            video = next((s for s in container.streams if s.type == "video"), None)
            if video is None:
                raise MergeError(409, f"{os.path.basename(path)} has no video stream.")
            audio = next((s for s in container.streams if s.type == "audio"), None)
            fps = float(video.average_rate or video.guessed_rate or 0)
            if video.duration is not None:
                seconds = float(video.duration * video.time_base)
            elif container.duration is not None:
                seconds = container.duration / 1_000_000
            else:
                seconds = (video.frames or 0) / fps if fps else 0.0
            if seconds <= 0:
                raise MergeError(409, f"{os.path.basename(path)} has no playable video.")
            return ClipInfo(
                path=path,
                video_codec=video.codec_context.name,
                width=video.codec_context.width,
                height=video.codec_context.height,
                fps=fps,
                pix_fmt=video.codec_context.pix_fmt or "",
                video_seconds=seconds,
                audio=((audio.codec_context.name, audio.codec_context.sample_rate, audio.codec_context.channels)
                       if audio else None),
            )
    except MergeError:
        raise
    except Exception as exc:  # av.error.* on a corrupt or unreadable file
        raise MergeError(409, f"Could not read {os.path.basename(path)}: {exc}") from exc


def check_compatible(infos: list[ClipInfo], labels: list[str]) -> None:
    """Refuse clips that can't be joined by copying, naming the first that differs."""
    first = infos[0]
    for info, label in zip(infos[1:], labels[1:], strict=True):
        if info.signature() != first.signature():
            raise MergeError(
                409,
                f"{label} ({info.describe()}) doesn't match {labels[0]} ({first.describe()}). "
                "Clips need the same size, frame rate and audio to be joined.",
            )


def _concat_list(infos: list[ClipInfo]) -> str:
    lines = ["ffconcat version 1.0"]
    for info in infos:
        escaped = info.path.replace("'", "'\\''")
        lines += [f"file '{escaped}'", f"outpoint {info.video_seconds:.6f}"]
    return "\n".join(lines) + "\n"


def merge_clips(paths: list[str], out_path: str, labels: list[str] | None = None, ffmpeg: str | None = None) -> float:
    """Join ``paths`` in order into ``out_path``. Returns the merged length in seconds."""
    if not paths:
        raise MergeError(400, "No clips to merge.")
    labels = labels or [os.path.basename(p) for p in paths]
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        raise MergeError(503, "ffmpeg isn't available on the server. Install it or set FASTVIDEO_FFMPEG_BIN.")

    infos = [probe_clip(p) for p in paths]
    check_compatible(infos, labels)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_out = f"{out_path}.tmp.mp4"
    list_fd, list_path = tempfile.mkstemp(suffix=".ffconcat", text=True)
    try:
        with os.fdopen(list_fd, "w", encoding="utf-8") as f:
            f.write(_concat_list(infos))
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy",
             "-movflags", "+faststart", tmp_out],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0:
            tail = " ".join(result.stderr.strip().splitlines()[-3:])
            raise MergeError(500, f"ffmpeg failed: {tail or 'unknown error'}")
        os.replace(tmp_out, out_path)  # atomic, so a reader never sees a half-written file
    except subprocess.TimeoutExpired as exc:
        raise MergeError(500, "ffmpeg took too long.") from exc
    finally:
        for leftover in (list_path, tmp_out):
            try:
                os.remove(leftover)
            except OSError:
                pass
    return sum(i.video_seconds for i in infos)


@dataclass(frozen=True)
class MergeResult:
    filename: str
    path: str
    clips: int
    seconds: float


def _safe_stem(name: str) -> str:
    return re.sub(r"[^\w.-]+", "-", name).strip("-.")[:60] or "scene"


def _label(job: Any) -> str:
    name = (getattr(job, "name", "") or "").strip()
    return f"'{name}'" if name else str(job.id)


def merge_jobs(jobs: list[Any], out_dir: str, name: str = "") -> MergeResult:
    """Join the finished videos of ``jobs`` (in the order given) into ``<out_dir>/merged``.

    Every job must be completed with a video on disk; otherwise nothing is made,
    so a scene missing a clip can't be mistaken for the whole thing.
    """
    if not jobs:
        raise MergeError(400, "No clips to merge.")
    paths: list[str] = []
    for job in jobs:
        status = getattr(job.status, "value", job.status)
        if status != "completed" or not job.output_path:
            raise MergeError(409, f"{_label(job)} hasn't finished ({status}), so the scene can't be merged yet.")
        if not os.path.isfile(job.output_path):
            raise MergeError(409, f"{_label(job)}'s video is missing from disk.")
        if Path(job.output_path).suffix.lower() not in VIDEO_EXTENSIONS:
            raise MergeError(409, f"{_label(job)}'s output isn't a video.")
        paths.append(job.output_path)

    filename = f"{_safe_stem(name)}-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    merged_dir = os.path.join(out_dir, MERGED_DIRNAME)
    out_path = os.path.join(merged_dir, filename)
    seconds = merge_clips(paths, out_path, labels=[_label(j) for j in jobs])
    return MergeResult(filename=filename, path=out_path, clips=len(paths), seconds=seconds)


def merged_path(out_dir: str, filename: str) -> str:
    """Path of a merged video by filename, refusing anything that isn't one."""
    if not _FILENAME_RE.match(filename):
        raise MergeError(404, "Merged video not found.")
    path = os.path.join(out_dir, MERGED_DIRNAME, filename)
    if not os.path.isfile(path):
        raise MergeError(404, "Merged video not found.")
    return path
