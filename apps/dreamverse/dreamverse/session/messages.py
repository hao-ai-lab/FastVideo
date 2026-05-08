"""Dataclasses for the prompt-submission pipeline queues."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptSubmission:
    prompt_id: str
    raw_prompt: str
    created_at_s: float


@dataclass
class ReadyPrompt:
    prompt: str
    source: str
    prompt_id: str | None = None
    fallback_used: bool = False
    seed_prompt_index: int | None = None
    loop_iteration: int | None = None
