"""Filesystem helpers shared by eval scripts.

Provides the prompt-sanitization, default filename convention, and
``(row, video_path) → eval-kwargs`` builder. Free functions, not a
class — :class:`fastvideo.eval.Evaluator` is the only stateful object
in the eval surface; loops live in user scripts.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Filesystem-unsafe characters mirrored from VideoGenerator's output-path
# sanitizer so on-disk filenames match what the generator writes.
_INVALID_CHARS = re.compile(r'[\\/:*?"<>|]')


def sanitize_prompt(prompt: str, max_len: int = 100) -> str:
    """Prompt → safe filename stem."""
    s = _INVALID_CHARS.sub("", prompt[:max_len]).strip().strip(".")
    return re.sub(r"\s+", " ", s) or "output"


def default_filename(row: dict, idx: int, ext: str = ".mp4") -> str:
    """``<sanitized-prompt>-<idx>.mp4`` — VBench-style."""
    return f"{sanitize_prompt(row['prompt'])}-{idx}{ext}"


def glob_videos(videos_dir: Path, row: dict, ext: str = ".mp4") -> list[Path]:
    """Find every generated video for *row*, sorted by trailing ``-<idx>``."""
    pattern = f"{sanitize_prompt(row['prompt'])}-*{ext}"
    files = list(videos_dir.glob(pattern))

    def _idx(p: Path) -> int:
        try:
            return int(p.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return sorted(files, key=_idx)


def build_eval_kwargs(row: dict, video_path: Path, *, fps: float = 24.0) -> dict[str, Any]:
    """Build evaluator kwargs from a sample row + a video on disk.

    Loads the video as ``(T,C,H,W)`` and adds the leading batch dim.
    Forwards ``prompt`` (as scalar ``text_prompt``) and
    ``auxiliary_info`` (as scalar dict) when present on the row —
    matches the one-sample-per-call contract that the evaluator and
    every metric assume.
    """
    from fastvideo.eval.io.video import load_video

    video = load_video(str(video_path))  # (T, C, H, W) in [0, 1]
    kwargs: dict[str, Any] = {
        "video": video.unsqueeze(0),  # (1, T, C, H, W)
        "fps": fps,
    }
    if "prompt" in row:
        kwargs["text_prompt"] = row["prompt"]
    aux = row.get("auxiliary_info")
    if aux:
        kwargs["auxiliary_info"] = aux
    return kwargs
