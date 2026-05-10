"""Shared helpers used by main.py, routes/, and session/."""
from __future__ import annotations

from datetime import datetime, timezone


def _main_print(level: str, message: str):
    print(f"[MAIN][{level}] {message}", flush=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


PROMPT_EXTENSION_FAILURE_USER_MESSAGE = ("Prompt extension failed for this request.")


def _resolve_generation_segment_cap(*, single_clip_mode: bool, cap: int) -> int:
    return 0 if single_clip_mode else cap
