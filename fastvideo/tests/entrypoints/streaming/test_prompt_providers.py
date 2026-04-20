# SPDX-License-Identifier: Apache-2.0
"""Tests for the LLM provider protocol + built-in adapters.

Real Cerebras/Groq HTTP calls are stubbed with a fake ``httpx`` module
inserted into ``sys.modules`` so the unit tests don't depend on
external API availability or paid keys.
"""
from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from fastvideo.entrypoints.streaming.prompt.providers import (
    CerebrasProvider,
    GroqProvider,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMTimeoutError,
)


@dataclass
class _FakeResponse:
    status_code: int
    payload: dict[str, Any]
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakeAsyncClient:

    def __init__(self, response_or_exc, *, captured: list) -> None:
        self._response_or_exc = response_or_exc
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, **kwargs):
        self._captured.append({"url": url, **kwargs})
        if isinstance(self._response_or_exc, Exception):
            raise self._response_or_exc
        return self._response_or_exc


def _install_fake_httpx(
    monkeypatch,
    *,
    response: _FakeResponse | None = None,
    exception: Exception | None = None,
) -> list[dict]:
    """Replace ``sys.modules['httpx']`` with a stub exposing the bits
    providers touch: ``AsyncClient``, ``HTTPError``, ``TimeoutException``.

    Returns a list the test can inspect to see what requests went out.
    """
    captured: list[dict] = []

    class _HTTPError(Exception):
        pass

    class _TimeoutException(_HTTPError):
        pass

    payload = response if exception is None else exception

    def _client_factory(*_args, **_kwargs):
        return _FakeAsyncClient(payload, captured=captured)

    stub = types.SimpleNamespace(
        AsyncClient=_client_factory,
        HTTPError=_HTTPError,
        TimeoutException=_TimeoutException,
    )
    monkeypatch.setitem(sys.modules, "httpx", stub)
    return captured


# ----------------------------------------------------------------------
# Cerebras
# ----------------------------------------------------------------------


class TestCerebrasProvider:

    def test_is_llm_provider(self):
        assert isinstance(CerebrasProvider(api_key="x"), LLMProvider)

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        provider = CerebrasProvider()
        with pytest.raises(LLMProviderError, match="CEREBRAS_API_KEY"):
            asyncio.run(provider.complete(
                LLMRequest(messages=[], model="m")))

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "from-env")
        provider = CerebrasProvider()
        assert provider.api_key == "from-env"

    def test_explicit_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "from-env")
        provider = CerebrasProvider(api_key="explicit")
        assert provider.api_key == "explicit"

    def test_success_path(self, monkeypatch):
        captured = _install_fake_httpx(
            monkeypatch,
            response=_FakeResponse(
                status_code=200,
                payload={
                    "choices": [{"message": {"content": "enhanced"}}],
                },
            ),
        )
        provider = CerebrasProvider(api_key="secret")
        result = asyncio.run(provider.complete(
            LLMRequest(
                messages=[LLMMessage(role="user", content="a fox")],
                model="gpt-oss-120b",
                max_tokens=64,
                temperature=0.7,
            )))
        assert result.content == "enhanced"
        assert result.provider == "cerebras"
        assert result.model == "gpt-oss-120b"
        # Verify the HTTP request was shaped correctly.
        body = captured[0]["json"]
        assert body["model"] == "gpt-oss-120b"
        assert body["messages"] == [{"role": "user", "content": "a fox"}]
        assert captured[0]["headers"]["Authorization"] == "Bearer secret"

    def test_http_error_wrapped(self, monkeypatch):
        # Stage the stub first, then raise that stub's own HTTPError so
        # the provider's ``except httpx.HTTPError`` catches it.
        stub = types.SimpleNamespace()

        class _HTTPError(Exception):
            pass

        class _TimeoutException(_HTTPError):
            pass

        stub.HTTPError = _HTTPError
        stub.TimeoutException = _TimeoutException
        stub.AsyncClient = lambda *a, **k: _FakeAsyncClient(
            _HTTPError("boom"), captured=[])
        monkeypatch.setitem(sys.modules, "httpx", stub)

        provider = CerebrasProvider(api_key="k")
        with pytest.raises(LLMProviderError, match="HTTP"):
            asyncio.run(provider.complete(
                LLMRequest(messages=[], model="m")))

    def test_timeout_wrapped(self, monkeypatch):
        _install_fake_httpx(monkeypatch)
        # Install timeout exception after stubbing.
        stub = sys.modules["httpx"]

        def raising_factory(*_a, **_kw):
            raise stub.TimeoutException("timed out")  # type: ignore[attr-defined]

        stub.AsyncClient = raising_factory  # type: ignore[attr-defined]
        provider = CerebrasProvider(api_key="k")
        with pytest.raises(LLMTimeoutError):
            asyncio.run(provider.complete(
                LLMRequest(messages=[], model="m")))

    def test_4xx_raises_provider_error(self, monkeypatch):
        _install_fake_httpx(
            monkeypatch,
            response=_FakeResponse(
                status_code=401,
                payload={},
                text="unauthorized",
            ),
        )
        provider = CerebrasProvider(api_key="k")
        with pytest.raises(LLMProviderError, match="401"):
            asyncio.run(provider.complete(
                LLMRequest(messages=[], model="m")))

    def test_empty_choices_raises(self, monkeypatch):
        _install_fake_httpx(
            monkeypatch,
            response=_FakeResponse(
                status_code=200, payload={"choices": []}),
        )
        provider = CerebrasProvider(api_key="k")
        with pytest.raises(LLMProviderError, match="no choices"):
            asyncio.run(provider.complete(
                LLMRequest(messages=[], model="m")))


# ----------------------------------------------------------------------
# Groq
# ----------------------------------------------------------------------


class TestGroqProvider:

    def test_is_llm_provider(self):
        assert isinstance(GroqProvider(api_key="x"), LLMProvider)

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        provider = GroqProvider()
        with pytest.raises(LLMProviderError, match="GROQ_API_KEY"):
            asyncio.run(provider.complete(
                LLMRequest(messages=[], model="m")))

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "from-env")
        provider = GroqProvider()
        assert provider.api_key == "from-env"

    def test_success_path(self, monkeypatch):
        captured = _install_fake_httpx(
            monkeypatch,
            response=_FakeResponse(
                status_code=200,
                payload={
                    "choices": [{"message": {"content": "groq out"}}],
                },
            ),
        )
        provider = GroqProvider(api_key="secret")
        result = asyncio.run(provider.complete(
            LLMRequest(
                messages=[LLMMessage(role="user", content="a deer")],
                model="llama-3.1-70b",
            )))
        assert result.content == "groq out"
        assert result.provider == "groq"
        assert captured[0]["headers"]["Authorization"] == "Bearer secret"
