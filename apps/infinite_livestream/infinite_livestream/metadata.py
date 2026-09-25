"""Small, GPU-independent display records carried by timed ID3 metadata."""

from __future__ import annotations

import json
from typing import Any

from .group_tag import parse_group_tag

PROMPT_PREVIEW = 180
ID3_DESCRIPTION = "infinite-livestream"


def clip_view(clip: dict[str, Any]) -> dict[str, Any]:
    """One queue entry, flattened for the page.

    Everything here comes from the clip's own `ClipInfo` plus the group tag the
    director wrote into its metadata, which the model echoes untouched.
    """
    tag = parse_group_tag(clip.get("metadata", "")) or {}
    # `prompt` is the upsampler's rewrite; the group tag keeps what the viewer
    # actually typed, and that is what the panel shows -- a viewer should
    # recognise their own words in the queue.
    original = tag.get("raw_prompt") or clip.get("prompt") or ""
    return {
        "clip_id": clip.get("clip_id", ""),
        "title": tag.get("title") or "",
        "author": tag.get("author") or "",
        "scene": tag.get("scene"),
        "scenes": tag.get("scenes"),
        "generated": bool(tag.get("generated")),
        # The author of filler is the stream itself; surfacing "auto" as a name
        # invites viewers to read it as another person's request.
        "author_label": ("" if tag.get("generated") else (tag.get("author") or "")),
        "seconds": clip.get("seconds"),
        "ready": bool(clip.get("ready")),
        "prompt": original[:PROMPT_PREVIEW],
        "expanded": (clip.get("prompt") or "")[:PROMPT_PREVIEW],
    }


def encode_id3(clip: dict[str, Any] | None) -> bytes:
    """Serialize one UTF-8 ID3v2.4 TXXX frame; timing belongs to its media packet."""
    record = json.dumps({"version": 1, "clip": clip}, ensure_ascii=False, separators=(",", ":"))
    payload = b"\x03" + ID3_DESCRIPTION.encode("ascii") + b"\x00" + record.encode("utf-8")

    def size(value: int) -> bytes:
        # ID3 uses four seven-bit bytes for both tag and v2.4 frame sizes.
        return bytes((value >> shift) & 0x7f for shift in (21, 14, 7, 0))

    frame = b"TXXX" + size(len(payload)) + b"\x00\x00" + payload
    return b"ID3\x04\x00\x00" + size(len(frame)) + frame


EMPTY_ID3 = encode_id3(None)
