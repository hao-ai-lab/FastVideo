# SPDX-License-Identifier: Apache-2.0
"""LLM provider protocol + DTOs used by the prompt enhancer.

Third-party users add a new provider by implementing
:class:`LLMProvider` and registering it with a prompt enhancer
instance. The shipped providers live in sibling modules
(``cerebras.py``, ``groq.py``) and each is ~100-200 LOC — the
provider layer is intentionally thin so the enhancer stays
provider-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMRequest:
    messages: list[LLMMessage]
    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    timeout_ms: int | None = None


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    latency_ms: float
    fallback_used: bool = False


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider fails a request.

    The enhancer treats this as a retryable error unless the concrete
    subclass overrides :attr:`retryable`.
    """

    retryable: bool = True


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM provider times out."""

    retryable: bool = True


@runtime_checkable
class LLMProvider(Protocol):
    """Provider interface every LLM adapter implements.

    Providers are async-first because every built-in implementation
    talks to an HTTP API. Synchronous providers can wrap their call in
    ``asyncio.to_thread`` internally.
    """

    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...


__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
]
