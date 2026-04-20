# SPDX-License-Identifier: Apache-2.0
"""Rewrite payload builder.

The UI's "rewrite seed prompts" flow asks the enhancer to produce a
batch of alternative prompts given one seed. This module packages the
seed + options into the payload the enhancer expects and unpacks the
response back into a typed :class:`RewriteResult`.

Separating this from :mod:`enhancer` keeps the enhancer provider-
agnostic; anything UI-specific (how many alternatives to request, how
to split the response, temperature) lives here.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastvideo.entrypoints.streaming.prompt.enhancer import PromptEnhancer


@dataclass
class RewriteOptions:
    count: int = 3
    """Number of alternative prompts to request."""
    temperature: float | None = None
    user_system_prompt_override: str | None = None


@dataclass
class RewriteResult:
    seed_prompt: str
    alternatives: list[str]
    provider: str
    model: str
    latency_ms: float
    fallback_used: bool = False


async def build_rewrite(
    enhancer: PromptEnhancer,
    seed_prompt: str,
    *,
    options: RewriteOptions | None = None,
) -> RewriteResult:
    """Run a rewrite op through the enhancer and return a typed result."""
    if not seed_prompt.strip():
        raise ValueError("rewrite seed prompt must be non-empty")
    options = options or RewriteOptions()
    response = await enhancer.rewrite(seed_prompt)
    alternatives = _split_response(response.content, limit=options.count)
    return RewriteResult(
        seed_prompt=seed_prompt,
        alternatives=alternatives,
        provider=response.provider,
        model=response.model,
        latency_ms=response.latency_ms,
        fallback_used=response.fallback_used,
    )


def _split_response(content: str, *, limit: int) -> list[str]:
    """Split the LLM response into discrete prompt candidates.

    The shipped system prompt instructs the model to emit one prompt
    per line; this function is forgiving about numbered lists or
    leading bullets so user-supplied system prompts don't break it.
    """
    import re

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    cleaned: list[str] = []
    leading_marker = re.compile(r"^(?:[-*•]\s*|\d+\s*[.)]\s*)+")
    for line in lines:
        stripped = leading_marker.sub("", line).strip()
        if stripped:
            cleaned.append(stripped)
    return cleaned[:max(1, limit)]


__all__ = [
    "RewriteOptions",
    "RewriteResult",
    "build_rewrite",
]
