"""The group tag: the JSON this app stores in a clip's metadata.

The director writes it at enqueue time and reads it back off the echo the
engine returns on every clip-referencing message, which is what lets a clip be
traced to the request that made it. `Director._enqueue_group` is the
authoritative writer.
"""

from __future__ import annotations

import json


def parse_group_tag(metadata: str) -> dict | None:
    """Read the tag back out of a clip's metadata echo, or None if absent."""
    try:
        tag = json.loads(metadata)
    except (TypeError, ValueError):
        return None
    if not isinstance(tag, dict) or "group_id" not in tag:
        return None
    return tag


def is_generated(clip: dict) -> bool:
    """Whether a clip is idle filler. Untagged clips count as viewer content."""
    tag = parse_group_tag(clip.get("metadata", ""))
    return bool(tag and tag.get("generated"))


def pick_next(clips: list[dict], ready_only: bool = True) -> dict | None:
    """The clip that should play next: viewer content first, then filler.

    Within each class queue order decides, so a group's scenes stay in
    sequence. `ready_only` false ranks clips that are still building.
    """
    pool = [c for c in clips if c.get("ready")] if ready_only else clips
    for clip in pool:
        if not is_generated(clip):
            return clip
    return pool[0] if pool else None


def viewer_insert_position(generation_clips: list[dict]) -> int | None:
    """Where a viewer clip enters the generation queue: ahead of filler.

    The index of the first filler clip, so viewer scenes land behind every
    viewer clip already waiting and ahead of filler, which just slides back.
    None when no filler waits and a plain append is already right.
    """
    for index, clip in enumerate(generation_clips):
        if is_generated(clip):
            return index
    return None
