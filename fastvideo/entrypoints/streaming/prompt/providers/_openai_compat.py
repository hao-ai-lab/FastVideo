# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP path for OpenAI-compatible ``/chat/completions`` providers.

Cerebras and Groq both expose the OpenAI chat-completions schema, so
the request shape, error mapping, and response decoding are identical
between them. This module centralizes that logic; the per-provider
modules stay thin (just defaults + env var wiring).
"""
from __future__ import annotations

import time

from fastvideo.entrypoints.streaming.prompt.providers.base import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)


async def complete_openai_compatible(
    *,
    api_key: str | None,
    api_key_hint: str,
    base_url: str,
    provider_name: str,
    request: LLMRequest,
) -> LLMResponse:
    """Issue a chat-completions call and decode the OpenAI response."""
    if not api_key:
        raise LLMProviderError(
            f"{provider_name} provider requires {api_key_hint} "
            "(or explicit api_key=...)",
            retryable=False,
        )
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dep
        raise LLMProviderError(
            f"{provider_name} provider requires httpx; install httpx",
            retryable=False,
        ) from exc

    timeout_s = (request.timeout_ms or 20000) / 1000.0
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": request.model,
                    "messages": [{
                        "role": m.role,
                        "content": m.content
                    } for m in request.messages],
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                },
            )
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(f"{provider_name} timed out after {timeout_s}s") from exc
    except httpx.HTTPError as exc:
        raise LLMProviderError(f"{provider_name} HTTP error: {exc}") from exc

    if response.status_code >= 400:
        # 5xx and 429 (rate-limit) are retryable: another provider may
        # succeed. 4xx (auth, bad-request, etc.) are client errors —
        # the enhancer should stop fallback traversal.
        retryable = (response.status_code >= 500 or response.status_code == 429)
        raise LLMProviderError(
            f"{provider_name} returned {response.status_code}: "
            f"{response.text[:200]}",
            retryable=retryable,
        )

    try:
        data = response.json()
    except Exception as exc:
        # Non-JSON body usually means a proxy / load-balancer error
        # page; leave it retryable so a fallback provider can try.
        raise LLMProviderError(f"{provider_name} returned non-JSON body: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise LLMProviderError(f"{provider_name} returned no choices")
    content = choices[0].get("message", {}).get("content") or ""

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return LLMResponse(
        content=content.strip(),
        provider=provider_name,
        model=request.model,
        latency_ms=latency_ms,
    )


__all__ = ["complete_openai_compatible"]
