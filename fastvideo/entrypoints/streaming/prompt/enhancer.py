# SPDX-License-Identifier: Apache-2.0
"""Provider-agnostic prompt orchestration for the streaming server.

Three operations the streaming server needs:

* ``enhance`` — polish a user prompt (add cinematic detail, fix syntax)
* ``auto_extend`` — generate a follow-on prompt for loop generation
* ``rewrite`` — rewrite a seed prompt for a user-directed rewrite flow

All three share the same orchestration: pick a provider in priority
order, submit an ``LLMRequest``, fall back to the next provider on
retryable errors, and surface a structured :class:`LLMResponse` back
to the caller.

System prompts are loaded from ``system_prompt_dir`` on construction
and can be hot-reloaded via :meth:`PromptEnhancer.reload_system_prompts`.
The streaming server's management endpoint calls that method in
response to a ``rewrite_seed_prompts_started`` frame.
"""
from __future__ import annotations

import enum
import os
from collections.abc import Sequence
from dataclasses import dataclass

from fastvideo.entrypoints.streaming.prompt.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class PromptOperation(enum.Enum):
    ENHANCE = "enhance"
    AUTO_EXTEND = "auto_extend"
    REWRITE = "rewrite"


@dataclass
class _SystemPrompts:
    enhance: str
    auto_extend: str
    rewrite: str


_DEFAULT_SYSTEM_PROMPTS = _SystemPrompts(
    enhance=("You are a prompt enhancer for cinematic video generation. Given "
             "a user prompt, produce an enhanced prompt that is more vivid, "
             "specific, and concrete. Keep the subject intact; add lighting, "
             "camera, and motion detail. Reply with just the enhanced prompt."),
    auto_extend=("You are a video continuation assistant. Given the current "
                 "sequence of prompts, produce one new prompt that naturally "
                 "continues the sequence. Reply with just the next prompt."),
    rewrite=("You are a creative prompt rewriter. Given a seed prompt, produce "
             "a set of alternative prompts that explore different angles, "
             "styles, and moods. Reply with one prompt per line."),
)


class PromptEnhancer:
    """Orchestrates prompt operations across a priority-ordered provider
    list with structured fallback + hot-reloadable system prompts.

    Usage::

        enhancer = PromptEnhancer(
            providers=[CerebrasProvider(), GroqProvider()],
            model="gpt-oss-120b",
            system_prompt_dir="/etc/fastvideo/prompts",
        )
        response = await enhancer.enhance("a fox running through snow")
    """

    def __init__(
        self,
        *,
        providers: Sequence[LLMProvider],
        model: str,
        timeout_ms: int = 20000,
        temperature: float = 0.7,
        max_tokens: int | None = 256,
        system_prompt_dir: str | None = None,
    ) -> None:
        if not providers:
            raise ValueError("PromptEnhancer requires at least one LLMProvider")
        self._providers = list(providers)
        self._model = model
        self._timeout_ms = timeout_ms
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt_dir = system_prompt_dir
        self._system_prompts = self._load_system_prompts()

    @property
    def providers(self) -> list[LLMProvider]:
        return list(self._providers)

    def register_provider(self, provider: LLMProvider, *, priority: int = -1) -> None:
        """Insert an additional provider. ``priority=0`` makes it primary;
        ``priority=-1`` (default) appends as a fallback."""
        if priority < 0:
            self._providers.append(provider)
        else:
            self._providers.insert(priority, provider)

    def reload_system_prompts(self) -> None:
        """Re-read the system prompt files from ``system_prompt_dir``.

        The streaming server exposes this via a management endpoint so
        operators can iterate on prompt templates without restarting
        workers.
        """
        self._system_prompts = self._load_system_prompts()
        logger.info("prompt enhancer: reloaded system prompts from %s", self._system_prompt_dir or "defaults")

    async def enhance(self, prompt: str) -> LLMResponse:
        return await self._run(
            PromptOperation.ENHANCE,
            system=self._system_prompts.enhance,
            user=prompt,
        )

    async def auto_extend(self, prior_prompts: Sequence[str]) -> LLMResponse:
        user = "\n".join(prior_prompts)
        return await self._run(
            PromptOperation.AUTO_EXTEND,
            system=self._system_prompts.auto_extend,
            user=user,
        )

    async def rewrite(self, seed_prompt: str) -> LLMResponse:
        return await self._run(
            PromptOperation.REWRITE,
            system=self._system_prompts.rewrite,
            user=seed_prompt,
        )

    async def _run(
        self,
        operation: PromptOperation,
        *,
        system: str,
        user: str,
    ) -> LLMResponse:
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            timeout_ms=self._timeout_ms,
        )
        last_error: LLMProviderError | None = None
        for idx, provider in enumerate(self._providers):
            try:
                response = await provider.complete(request)
                if idx > 0:
                    # Rebuild so the response records that a fallback
                    # was used (without reaching into provider state).
                    response = LLMResponse(
                        content=response.content,
                        provider=response.provider,
                        model=response.model,
                        latency_ms=response.latency_ms,
                        fallback_used=True,
                    )
                return response
            except LLMProviderError as exc:
                logger.warning("prompt %s: provider %s failed: %s; trying next", operation.value, provider.name, exc)
                last_error = exc
                if not exc.retryable:
                    break
        assert last_error is not None
        raise last_error

    def _load_system_prompts(self) -> _SystemPrompts:
        if not self._system_prompt_dir:
            return _DEFAULT_SYSTEM_PROMPTS
        return _SystemPrompts(
            enhance=_read_prompt(self._system_prompt_dir, "enhance.txt", _DEFAULT_SYSTEM_PROMPTS.enhance),
            auto_extend=_read_prompt(self._system_prompt_dir, "auto_extend.txt", _DEFAULT_SYSTEM_PROMPTS.auto_extend),
            rewrite=_read_prompt(self._system_prompt_dir, "rewrite.txt", _DEFAULT_SYSTEM_PROMPTS.rewrite),
        )


def _read_prompt(dirname: str, filename: str, default: str) -> str:
    path = os.path.join(dirname, filename)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    return content or default


__all__ = ["PromptEnhancer", "PromptOperation"]
