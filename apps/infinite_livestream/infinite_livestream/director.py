"""The director: viewer prompts in, tagged scene groups on the engine's queue.

One chat prompt becomes one *scene group*: the upsampler expands it into 1..N
self-contained scenes -- a single shot, or a chunked short story -- which the
director enqueues contiguously. It is also the playout brain: `run_playout`
curates the front of the playout queue with `move` so the engine's next
autoplay is already the right clip.

Rules that keep it coherent:

  * It is the queue's only writer. The viewer worker (`run`) and the idle
    filler (`run_idle`) serialise their enqueues through one lock, so groups
    can never interleave.
  * A group is enqueued only when the whole group fits, so it cannot get
    stuck half-in. Capacities come from the engine's `state_update`, never
    from constants here.
  * Viewer prompts outrank filler and stay first-come-first-served among
    themselves: viewer groups insert ahead of waiting filler and behind
    waiting viewer clips, the playout loop pops one built filler when a full
    playout queue blocks a viewer's build, and the idle filler stands down
    whenever viewer work is pending.

Every scene carries its group's identity in the clip metadata, which the
engine echoes back on every message referencing that clip. That is what lets
"scene 2/3 of Neon Alley by viewer_42" be reconstructed from a `clip_started`
alone, and what marks filler as evictable later.

Viewer prompts pass moderation before the upsampler; the curated idle list
does not need it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable, Sequence

from .chat import ChatPrompt
from .group_tag import is_generated, parse_group_tag, pick_next, viewer_insert_position
from .engine import Engine
from .moderator import Moderator
from .upsampler import PromptUpsampler, SceneGroup

logger = logging.getLogger(__name__)

# Prompts waiting for upsampling+enqueue before new ones are turned away.
# Depth here is viewer wait time, and a backlog on top of a full generation
# queue serves nobody.
_PENDING_LIMIT = 24

# Enqueue retry cadence while the model refuses (reconnect mid-command, ...).
_RETRY_DELAY_S = 3.0

# How often the idle filler re-checks whether the queue wants topping up.
_IDLE_POLL_S = 3.0

# How often the playout loop re-checks. The broadcasts keep the mirrors
# fresh; polling them is what survives a missed message.
_PLAYOUT_POLL_S = 0.5


class Director:
    """Consume chat prompts; keep the fast-h3 queue fed with scene groups."""

    def __init__(
        self,
        link: Engine,
        upsampler: PromptUpsampler,
        moderator: Moderator,
        cooldown_s: float,
        idle_prompts: Sequence[str] = (),
        idle_queue_target: int = 0,
        on_reject: Callable[[str, str], None] | None = None,
    ) -> None:
        self._link = link
        self._on_reject = on_reject
        self._upsampler = upsampler
        self._moderator = moderator
        self._cooldown_s = cooldown_s
        self._idle_prompts = list(idle_prompts)
        random.shuffle(self._idle_prompts)
        self._idle_index = 0
        self._idle_target = idle_queue_target
        self._pending: asyncio.Queue[ChatPrompt] = asyncio.Queue(_PENDING_LIMIT)
        self._last_accepted: dict[str, float] = {}  # author -> monotonic
        self._enqueue_lock = asyncio.Lock()
        link.add_listener(self._on_model_message)

    # -------------------------------------------------------- chat intake

    def cooldown_remaining(self, author: str) -> float:
        """Seconds until *author* may send again; 0 when they may send now.

        Asked by the web app before it accepts a POST, so a rate-limited
        viewer is stopped in their own browser rather than told afterwards in
        a chat feed everyone else can read.
        """
        last = self._last_accepted.get(author)
        if last is None:
            return 0.0
        return max(0.0, self._cooldown_s - (time.monotonic() - last))

    def _reject(self, prompt: ChatPrompt, reason: str) -> None:
        """Drop one prompt, and make sure the viewer hears about it.

        Every rejection below used to be log-only, while the web app had
        already answered the POST with `ok` and echoed the prompt into chat --
        so a viewer watched their request appear and then quietly die. The
        component that can say no is downstream of the acknowledgement, which
        is why it has to report back rather than return a status.
        """
        logger.info("[director] dropped from %s@%s (%s): %s", prompt.author, prompt.source, reason, prompt.text)
        if self._on_reject is None:
            return
        try:
            self._on_reject(prompt.author, reason)
        except Exception:  # noqa: BLE001 -- telling the viewer must not kill the loop
            logger.exception("[director] reject callback failed")

    def submit(self, prompt: ChatPrompt) -> None:
        """Accept one chat prompt (called synchronously by chat sources)."""
        now = time.monotonic()
        last = self._last_accepted.get(prompt.author)
        if last is not None and now - last < self._cooldown_s:
            self._reject(prompt, f"one prompt every {self._cooldown_s:.0f}s; "
                         f"{self._cooldown_s - (now - last):.0f}s left")
            return
        try:
            self._pending.put_nowait(prompt)
        except asyncio.QueueFull:
            self._reject(prompt, f"the backlog is full ({_PENDING_LIMIT} waiting)")
            return
        self._last_accepted[prompt.author] = now
        logger.info(
            "[director] accepted from %s@%s: %s",
            prompt.author,
            prompt.source,
            prompt.text,
        )

    # ------------------------------------------------- viewer prompt loop

    def _viewer_clips_queued(self) -> int:
        """Viewer clips across both queues (anything not tagged filler)."""
        return sum(1 for clip in self._link.generation_clips + self._link.playout_clips if not is_generated(clip))

    async def run(self) -> None:
        """Moderate, upsample, and enqueue pending prompts, one group at a time."""
        while True:
            prompt = await self._pending.get()
            try:
                # Dropped now, before it costs a moderation and an LLM call.
                # Capacity comes from the engine, never from a constant.
                if self._viewer_clips_queued() >= self._link.playout_capacity:
                    self._reject(
                        prompt, f"{self._viewer_clips_queued()} viewer clips already queued "
                        f"(budget {self._link.playout_capacity})")
                    continue
                verdict = await self._moderator.review(prompt.text)
                if verdict is not None:
                    self._reject(prompt, verdict)
                    continue
                group = await self._upsampler.upsample(
                    raw_prompt=prompt.text,
                    author=prompt.author,
                    source=prompt.source,
                    min_seconds=self._link.min_seconds,
                    max_seconds=self._link.max_seconds,
                )
                await self._enqueue_group(group)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "[director] failed to process prompt from %s: %s",
                    prompt.author,
                    error,
                )

    # -------------------------------------------------------- idle filler

    async def run_idle(self) -> None:
        """Keep the queue topped up with generated clips while chat is quiet.

        One clip per group, on purpose: single-scene fillers are the finest
        eviction granularity, and popping one never truncates a story.
        """
        if self._idle_target <= 0:
            logger.info("[director] idle filler disabled (target 0)")
            return
        logger.info(
            "[director] idle filler: %d prompts, queue target %d",
            len(self._idle_prompts),
            self._idle_target,
        )
        while True:
            await asyncio.sleep(_IDLE_POLL_S)
            # May be empty after a switch to a preset with no idle prompts;
            # keep polling so a later switch revives it without a restart.
            if not self._idle_prompts:
                continue
            # The configured target self-clamps under the deployment's live
            # playout capacity: filler must never be what fills the playout
            # queue to the brim, because a full playout queue pauses builds
            # (leave at least one slot's headroom for a viewer clip to land).
            target = min(self._idle_target, max(1, self._link.playout_capacity - 1))
            if (not self._pending.empty() or not self._link.connected
                    or self._link.generation_queued + self._link.playout_queued >= target):
                continue
            text = self._idle_prompts[self._idle_index % len(self._idle_prompts)]
            self._idle_index += 1
            try:
                group = await self._upsampler.upsample(
                    raw_prompt=text,
                    author="auto",
                    source="idle",
                    min_seconds=self._link.min_seconds,
                    max_seconds=self._link.max_seconds,
                    generated=True,
                    max_chunks=1,
                )
                # A viewer prompt that arrived while the LLM ran outranks the
                # filler; drop this group rather than making the viewer wait.
                if self._pending.empty():
                    await self._enqueue_group(group)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("[director] idle fill failed: %s", error)

    # ------------------------------------------------------------- playout

    async def run_playout(self) -> None:
        """Curate the playout queue's front so autoplay always starts right.

        The engine chains the playout front the instant the stream idles, so
        nothing here sends an explicit play; it keeps the front correct
        instead. Reordering happens while a clip plays, ahead of the moment it
        matters.
        """
        while True:
            await asyncio.sleep(_PLAYOUT_POLL_S)
            if not self._link.connected:
                continue
            await self._relieve_build_backpressure()
            clips = self._link.playout_clips
            desired = pick_next(clips)
            if desired is None or clips[0]["clip_id"] == desired["clip_id"]:
                continue
            await self._link.send_command("move", {"clip_id": desired["clip_id"], "position": 0})
            # Let the resulting queue_update land before re-evaluating.
            await asyncio.sleep(_PLAYOUT_POLL_S)

    async def _relieve_build_backpressure(self) -> None:
        """Pop one playout filler when built fillers block a viewer's build.

        Generation pauses while the playout queue is full. When what fills it
        is idle filler and a viewer clip waits to build, the newest filler is
        the right thing to lose — one per tick, so a draining queue gets
        every chance to make room by playing instead.
        """
        if self._link.playout_queued < self._link.playout_capacity:
            return
        viewer_waiting = any(not is_generated(clip) for clip in self._link.generation_clips)
        if not viewer_waiting:
            return
        for clip in reversed(self._link.playout_clips):
            if is_generated(clip):
                reply = await self._link.send_command("pop", {"clip_id": clip["clip_id"]})
                if isinstance(reply, dict) and "clip" in reply:
                    logger.info(
                        "[director] popped playout filler %s to unblock a "
                        "viewer build",
                        clip["clip_id"][:8],
                    )
                return

    # ---------------------------------------------------------- enqueueing

    async def _enqueue_group(self, group: SceneGroup) -> None:
        """Put one group on the model's generation queue, or drop it and say why.

        Viewer groups enter *ahead of waiting filler and behind waiting
        viewer clips* (`viewer_insert_position`), so viewer requests stay
        first-come-first-served and idle filler just slides back — no
        popping, no waste. Filler groups append. When the generation queue
        cannot fit the group even after dropping the filler waiting in it,
        the group is dropped with the queues intact — a backlog full of
        viewer content takes no more, rather than stalling every later
        prompt behind a wait.
        """
        scene_count = len(group.scenes)
        async with self._enqueue_lock:
            free = self._link.generation_capacity - self._link.generation_queued
            if free < scene_count and not group.generated:
                evictable = sum(1 for clip in self._link.generation_clips if is_generated(clip))
                if free + evictable >= scene_count:
                    await self._evict_generation_fillers(scene_count - free)
                    await asyncio.sleep(0.3)  # let the pops' queue_update land
                    free = (self._link.generation_capacity - self._link.generation_queued)
            if free < scene_count:
                logger.warning(
                    "[director] no room in the generation queue for %s "
                    "(%d scenes, %d free); dropping the group",
                    group.group_id,
                    scene_count,
                    free,
                )
                return

            position = (None if group.generated else viewer_insert_position(self._link.generation_clips))
            for index, scene in enumerate(group.scenes, start=1):
                metadata = json.dumps(
                    {
                        "group_id": group.group_id,
                        "title": group.title[:120],
                        "scene": index,
                        "scenes": scene_count,
                        "author": group.author,
                        "source": group.source,
                        "generated": group.generated,
                        # Truncated so the whole blob stays well under fast-h3's
                        # 2000-char metadata cap.
                        "raw_prompt": group.raw_prompt[:400],
                    },
                    ensure_ascii=False,
                )
                payload = {
                    "prompt": scene.prompt,
                    "metadata": metadata,
                    "seconds": scene.seconds,
                }
                if position is not None:
                    # Consecutive positions keep the group contiguous and in
                    # scene order, ahead of the filler it displaced.
                    payload["position"] = position + index - 1
                while True:
                    reply = await self._link.send_command("enqueue", payload)
                    if isinstance(reply, dict) and "clip" in reply:
                        clip = reply["clip"]
                        logger.info(
                            "[director] queued %s scene %d/%d as %s (%.1fs, seed %s)%s",
                            group.group_id,
                            index,
                            scene_count,
                            clip["clip_id"][:8],
                            clip["seconds"],
                            clip["seed"],
                            " [auto]" if group.generated else "",
                        )
                        break
                    # A bodyless reply means refused; the engine already
                    # logged why. Wait and retry.
                    logger.warning(
                        "[director] enqueue of %s scene %d/%d refused; retrying in %.0fs",
                        group.group_id,
                        index,
                        scene_count,
                        _RETRY_DELAY_S,
                    )
                    await asyncio.sleep(_RETRY_DELAY_S)

    async def _evict_generation_fillers(self, needed: int) -> int:
        """Pop up to `needed` filler clips from the generation queue.

        Capacity relief only — order needs no eviction now that viewer
        groups insert ahead of filler positionally. Newest-queued first, and
        only clips tagged `generated: true`. Returns how many pops succeeded.
        """
        popped = 0
        for clip in reversed(self._link.generation_clips):
            if popped >= needed:
                break
            if not is_generated(clip):
                continue
            reply = await self._link.send_command("pop", {"clip_id": clip["clip_id"]})
            if isinstance(reply, dict) and "clip" in reply:
                popped += 1
                logger.info(
                    "[director] evicted waiting filler %s for a viewer group",
                    clip["clip_id"][:8],
                )
        return popped

    # ----------------------------------------------------- announcements

    def _on_model_message(self, kind: str, data: dict) -> None:
        """Narrate group playback from clip messages alone (via metadata)."""
        clip = data.get("clip") if isinstance(data, dict) else None
        if not isinstance(clip, dict):
            return
        tag = parse_group_tag(clip.get("metadata", ""))
        label = (f"'{tag['title']}' scene {tag['scene']}/{tag['scenes']} "
                 f"(by {tag['author']}@{tag['source']})" +
                 (" [auto]" if tag.get("generated") else "") if tag else f"clip {clip.get('clip_id', '?')[:8]}")
        if kind == "clip_started":
            logger.info("[now playing] %s", label)
        elif kind == "clip_finished":
            logger.info("[finished] %s", label)
        elif kind == "clip_failed":
            logger.error(
                "[director] build failed for %s: %s — the queue moves on",
                label,
                data.get("reason"),
            )
