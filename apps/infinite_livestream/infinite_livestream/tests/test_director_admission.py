"""Whether a viewer's prompt survives, and whether they are told when it does not.

This path had no tests, which is how a silent drop reached a live stream: the
web app answers the POST with `ok` and echoes the prompt into chat, then the
director -- downstream of that acknowledgement -- can still refuse it. Anything
that refuses here has to report back, or the viewer watches their request
appear and then vanish.

No GPU, no network: the engine, upsampler and moderator are stubs, because what
is under test is the admission decision, not what happens after it.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from infinite_livestream.chat import ChatPrompt
from infinite_livestream.director import Director


class FakeEngine:
    """Just enough of `Engine` for the director's capacity checks."""

    def __init__(self, playout_capacity: int = 10) -> None:
        self.generation_clips: list[dict] = []
        self.playout_clips: list[dict] = []
        self.playout_capacity = playout_capacity
        self.generation_capacity = 20
        self.playout_queued = 0
        self.generation_queued = 0
        self.min_seconds, self.max_seconds = 5.167, 14.375
        self.connected = True
        self.commands: list[tuple[str, dict]] = []

    def add_listener(self, listener) -> None:
        pass

    async def send_command(self, command: str, data: dict):
        self.commands.append((command, data))
        return {"clip": {"clip_id": "x" * 12, "seconds": 14.4, "seed": 1}}


class FakeModerator:
    enabled = True

    def __init__(self, verdict: str | None = None) -> None:
        self.verdict = verdict

    async def review(self, text: str) -> str | None:
        return self.verdict


def make_director(rejections: list[tuple[str, str]], *, cooldown_s: float = 10.0,
                  engine: FakeEngine | None = None, moderator: FakeModerator | None = None) -> Director:
    # Deliberate test doubles: what is under test is the admission decision,
    # which touches none of the real collaborators' behaviour.
    return Director(
        cast("Any", engine or FakeEngine()),
        upsampler=cast("Any", None),
        moderator=cast("Any", moderator or FakeModerator()),
        cooldown_s=cooldown_s,
        idle_prompts=(),
        idle_queue_target=0,
        on_reject=lambda author, reason: rejections.append((author, reason)),
    )


def prompt(author: str = "ada", text: str = "a lighthouse keeper") -> ChatPrompt:
    return ChatPrompt(source="web", author=author, text=text, command="!prompt")


def test_first_prompt_is_accepted() -> None:
    rejections: list[tuple[str, str]] = []
    director = make_director(rejections)
    director.submit(prompt())
    assert rejections == []
    assert director._pending.qsize() == 1


def test_cooldown_drop_is_reported_to_the_viewer() -> None:
    """The bug that reached production: accepted, echoed, then silently gone."""
    rejections: list[tuple[str, str]] = []
    director = make_director(rejections, cooldown_s=30.0)
    director.submit(prompt())
    director.submit(prompt())
    assert director._pending.qsize() == 1, "the second must not be queued"
    assert len(rejections) == 1, "and the viewer must be told"
    author, reason = rejections[0]
    assert author == "ada"
    assert "s left" in reason, f"the reason should say how long to wait, got {reason!r}"


def test_cooldown_is_per_author() -> None:
    """Two people must not silence each other."""
    rejections: list[tuple[str, str]] = []
    director = make_director(rejections, cooldown_s=30.0)
    director.submit(prompt(author="ada"))
    director.submit(prompt(author="grace"))
    assert rejections == []
    assert director._pending.qsize() == 2


def test_backlog_full_is_reported() -> None:
    rejections: list[tuple[str, str]] = []
    director = make_director(rejections, cooldown_s=0.0)
    for i in range(64):
        director.submit(prompt(author=f"viewer{i}"))
    assert rejections, "a full backlog must be reported, not swallowed"
    assert any("backlog" in reason for _, reason in rejections)


def test_moderation_rejection_is_reported() -> None:
    rejections: list[tuple[str, str]] = []
    director = make_director(rejections, moderator=FakeModerator(verdict="flagged: violence"))
    director.submit(prompt())
    asyncio.run(_drain_once(director))
    assert rejections == [("ada", "flagged: violence")]


def test_viewer_budget_full_is_reported() -> None:
    """A queue already full of viewer content refuses more, and says so."""
    rejections: list[tuple[str, str]] = []
    engine = FakeEngine(playout_capacity=2)
    engine.playout_clips = [{"metadata": "{}", "clip_id": "a"}, {"metadata": "{}", "clip_id": "b"}]
    director = make_director(rejections, engine=engine)
    director.submit(prompt())
    asyncio.run(_drain_once(director))
    assert rejections and "already queued" in rejections[0][1]


async def _drain_once(director: Director) -> None:
    """Run the prompt loop just long enough to process what is pending."""
    task = asyncio.create_task(director.run())
    for _ in range(50):
        await asyncio.sleep(0)
        if director._pending.empty():
            break
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_cooldown_remaining_reports_the_wait() -> None:
    """The web app asks this before accepting, so a rate-limited viewer is
    stopped in their own browser rather than announced to the shared feed."""
    director = make_director([], cooldown_s=30.0)
    assert director.cooldown_remaining("ada") == 0.0
    director.submit(prompt(author="ada"))
    remaining = director.cooldown_remaining("ada")
    assert 25.0 < remaining <= 30.0
    assert director.cooldown_remaining("grace") == 0.0, "and it is per author"
