# SPDX-License-Identifier: Apache-2.0
"""Groq LLM provider (OpenAI-compatible chat endpoint)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from fastvideo.entrypoints.streaming.prompt.providers.base import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class GroqProvider:
    """Groq inference adapter.

    API surface is identical to Cerebras (OpenAI-compatible); the two
    providers differ primarily in base URL + model id conventions.
    """

    api_key: str | None = None
    base_url: str = _DEFAULT_BASE_URL
    name: str = "groq"

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("GROQ_API_KEY")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise LLMProviderError("groq provider requires GROQ_API_KEY (or explicit api_key=...)")

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dep
            raise LLMProviderError("groq provider requires httpx; install httpx") from exc

        timeout_s = ((request.timeout_ms or 20000) / 1000.0)
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
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
            raise LLMTimeoutError(f"groq timed out after {timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"groq HTTP error: {exc}") from exc

        if response.status_code >= 400:
            raise LLMProviderError(f"groq returned {response.status_code}: "
                                   f"{response.text[:200]}")

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMProviderError("groq returned no choices")
        content = choices[0].get("message", {}).get("content") or ""

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return LLMResponse(
            content=content.strip(),
            provider=self.name,
            model=request.model,
            latency_ms=latency_ms,
        )


__all__ = ["GroqProvider"]
