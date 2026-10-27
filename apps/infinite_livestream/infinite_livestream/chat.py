"""Viewer prompts, typed into the page that plays the stream.

The page is the only way in, so this is fed directly in-process by `webapp.py`
rather than polling a platform. `submit` is called from a request handler and
never awaits: a full queue drops the message and tells the viewer, which is
better than stalling the web server.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger("infinite_livestream.chat")

# Small on purpose: a deep queue would let a burst of typing commit the stream
# to minutes of stale prompts.
QUEUE_SIZE = 32


@dataclass(frozen=True)
class ChatPrompt:
    """One accepted message, command word stripped."""

    source: str
    author: str
    text: str
    command: str = ""
    received_at: float = field(default_factory=time.monotonic)


def match_command(message: str, commands: Sequence[str]) -> tuple[str, str] | None:
    """Match a message against command words: `(command, text)` or None.

    Case-insensitive, and a bare command with no text is ignored.
    """
    stripped = message.strip()
    lowered = stripped.lower()
    for command in commands:
        if not lowered.startswith(command.lower()):
            continue
        remainder = stripped[len(command):]
        if remainder and not remainder[0].isspace():
            continue  # "!promptfoo" is not "!prompt foo"
        text = remainder.strip()
        if text:
            return command, text
    return None


class WebChat:
    """Prompts submitted through the page's chat box."""

    name = "web"

    def __init__(self, command: str = "!prompt") -> None:
        self._command = command
        self._queue: asyncio.Queue[ChatPrompt] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._dropped = 0

    def submit(self, author: str, text: str, command: str | None = None) -> bool:
        """Accept one message. True when it was queued, False when dropped."""
        text = text.strip()
        if not text:
            return False
        matched = match_command(text, (self._command, ))
        word, body = matched if matched else (self._command, text)
        prompt = ChatPrompt(source=self.name, author=author or "viewer", text=body, command=command or word)
        try:
            self._queue.put_nowait(prompt)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("[chat] queue full, dropped prompt from %s (%d total)", prompt.author, self._dropped)
            return False
        return True

    async def run(self, on_prompt: Callable[[ChatPrompt], None]) -> None:
        logger.info("[chat] ready (queue %d)", QUEUE_SIZE)
        while True:
            on_prompt(await self._queue.get())
