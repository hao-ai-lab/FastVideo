"""Chat, queue, and service state stay independent of viewers' playback clocks."""

from __future__ import annotations

from infinite_livestream.webapp import DemoState


def clip(clip_id: str = "abcdef123456", prompt: str = "a lighthouse keeper", *, generated: bool = False,
         scene: int | None = None, scenes: int | None = None) -> dict:
    import json
    meta = {"group_id": "g1", "title": prompt, "author": "viewer", "generated": generated, "raw_prompt": prompt}
    if scene is not None:
        meta |= {"scene": scene, "scenes": scenes}
    return {"clip_id": clip_id, "prompt": prompt, "metadata": json.dumps(meta), "frames": 345,
            "seconds": 14.375, "seed": 1, "ready": True}


def test_queue_update_replaces_both_queues() -> None:
    state = DemoState()
    state.on_message("queue_update", {"generation": [clip("a")], "playout": [clip("b"), clip("c")]})
    assert [c["clip_id"] for c in state.generation] == ["a"]
    assert [c["clip_id"] for c in state.playout] == ["b", "c"]
    # Replacement, not accumulation: a queue that empties must render empty.
    state.on_message("queue_update", {"generation": [], "playout": []})
    assert state.generation == [] and state.playout == []


def test_generating_is_the_generation_front() -> None:
    """Builds consume the queue front-first, so the front is what is in flight."""
    state = DemoState()
    assert state.generating is None
    state.on_message("queue_update", {"generation": [clip("a"), clip("b")], "playout": []})
    generating = state.generating
    assert generating is not None and generating["clip_id"] == "a"


def test_playout_events_do_not_define_a_viewers_playback_position() -> None:
    state = DemoState()
    before = state.snapshot()
    state.on_message("clip_started", {"clip": clip("a")})
    state.on_message("clip_finished", {"clip": clip("a"), "seconds_sent": 14.4})
    assert state.snapshot() == before


def test_only_filler_is_announced_in_chat_and_once_per_group() -> None:
    """Viewer submissions are echoed by the POST handler, so only filler here.

    And one line per group, not per scene: a six-scene story is still one
    thing somebody asked for.
    """
    state = DemoState()
    state.on_message("clip_queued", {"clip": clip("v", "viewer idea", generated=False)})
    assert list(state.chat) == []
    for scene in (1, 2, 3):
        state.on_message("clip_queued", {"clip": clip(f"f{scene}", "filler idea", generated=True,
                                                      scene=scene, scenes=3)})
    assert [c["author"] for c in state.chat] == ["filler"]


def test_failed_viewer_clips_are_reported_but_filler_is_not() -> None:
    state = DemoState()
    state.on_message("clip_failed", {"clip": clip("f", generated=True), "reason": "boom"})
    assert list(state.chat) == []
    state.on_message("clip_failed", {"clip": clip("v", "viewer idea", generated=False), "reason": "boom"})
    assert [c["kind"] for c in state.chat] == ["error"]


def test_snapshot_carries_everything_the_page_reads() -> None:
    state = DemoState()
    state.on_message("state_update", {"playing": False, "generation_queued": 1, "generation_capacity": 20,
                                      "playout_queued": 2, "playout_capacity": 10, "clips_played": 7,
                                      "width": 1344, "height": 768})
    snap = state.snapshot()
    assert set(snap) == {"connected", "generating",
                         "generation", "playout", "stats", "chat"}
    assert snap["stats"]["clips_played"] == 7
