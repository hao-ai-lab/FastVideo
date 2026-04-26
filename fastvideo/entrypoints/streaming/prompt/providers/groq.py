# SPDX-License-Identifier: Apache-2.0
"""Groq LLM provider (OpenAI-compatible chat endpoint)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from fastvideo.entrypoints.streaming.prompt.providers._openai_compat import (
    complete_openai_compatible, )
from fastvideo.entrypoints.streaming.prompt.providers.base import (
    LLMRequest,
    LLMResponse,
)

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_API_KEY_ENV = "GROQ_API_KEY"


@dataclass
class GroqProvider:
    """Groq inference adapter.

    Identical wire format to :class:`CerebrasProvider`; both go through
    :func:`complete_openai_compatible`. The two providers differ only
    in base URL, env var, and model id conventions.
    """

    api_key: str | None = None
    base_url: str = _DEFAULT_BASE_URL
    name: str = "groq"

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get(_API_KEY_ENV)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await complete_openai_compatible(
            api_key=self.api_key,
            api_key_hint=_API_KEY_ENV,
            base_url=self.base_url,
            provider_name=self.name,
            request=request,
        )


__all__ = ["GroqProvider"]
