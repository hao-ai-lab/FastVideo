# SPDX-License-Identifier: Apache-2.0
"""Prompt pipeline for the streaming server.

* :mod:`providers` — LLM backend abstraction + built-in adapters
* :mod:`enhancer` — provider-agnostic enhance / auto-extend / rewrite
  operations on top of the provider layer

All of this is optional; the streaming server runs fine without it
(PR 7.5's skeleton never invokes the enhancer). When the operator
enables ``ServeConfig.streaming.prompt.enabled``, the server routes
each ``session_init_v2`` curated prompt through ``enhance`` before the
first segment.
"""
from fastvideo.entrypoints.streaming.prompt.enhancer import (
    PromptEnhancer,
    PromptOperation,
)
from fastvideo.entrypoints.streaming.prompt.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)

__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
    "PromptEnhancer",
    "PromptOperation",
]
