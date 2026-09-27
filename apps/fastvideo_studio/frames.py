# SPDX-License-Identifier: Apache-2.0
"""Save the final frame of a finished clip, so the next clip can start from it.

Kept apart from ``server.py`` because it needs nothing from FastVideo -- only
PyAV -- and so can be exercised without a GPU.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


class FrameError(Exception):
    """A request that can't be satisfied, carrying the HTTP status to report."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def save_last_frame(video_path: str, out_path: str) -> None:
    """Decode ``video_path`` and write its final frame to ``out_path`` as a PNG.

    Clips are a few hundred frames, so decoding through to the end is simpler
    and more reliable than seeking (which lands on keyframes, not the last frame).
    """
    import av

    tmp_path = f"{out_path}.tmp.png"
    with av.open(video_path) as container:
        last = None
        for frame in container.decode(container.streams.video[0]):
            last = frame
    if last is None:
        raise ValueError("The video has no frames.")
    last.to_image().save(tmp_path)
    os.replace(tmp_path, out_path)  # atomic, so a reader never sees a half-written file


def last_frame_for_job(job: Any, upload_dir: str) -> str:
    """Path of a PNG of the last frame of ``job``'s video, extracting it if needed.

    Raises :class:`FrameError` with the status the API should return.
    """
    status = getattr(job.status, "value", job.status)
    if status != "completed" or not job.output_path:
        raise FrameError(404, "No output available for this job")
    if not os.path.isfile(job.output_path):
        raise FrameError(404, "Output file not found on disk")
    if Path(job.output_path).suffix.lower() not in VIDEO_EXTENSIONS:
        raise FrameError(400, "This job's output is not a video")
    if not upload_dir:
        raise FrameError(503, "Upload directory not configured")

    out_dir = os.path.join(upload_dir, "last_frames")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, f"last_frame_{job.id}.png"))
    # Reuse an earlier extraction unless the video has been rewritten since.
    if not (os.path.isfile(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(job.output_path)):
        try:
            save_last_frame(job.output_path, out_path)
        except Exception as e:
            raise FrameError(500, f"Could not read the last frame: {e}") from e
    return out_path
