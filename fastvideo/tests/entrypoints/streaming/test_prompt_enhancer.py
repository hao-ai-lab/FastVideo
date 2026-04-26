# SPDX-License-Identifier: Apache-2.0
"""Tests for the provider-agnostic prompt enhancer."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from fastvideo.entrypoints.streaming.prompt import (
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    PromptEnhancer,
)


@dataclass
class _StaticProvider:
    name: str
    content: str

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=self.content,
            provider=self.name,
            model=request.model,
            latency_ms=1.0,
        )


@dataclass
class _FailingProvider:
    name: str
    message: str = "boom"
    retryable: bool = True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError(self.message, retryable=self.retryable)


class TestConstruction:

    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError):
            PromptEnhancer(providers=[], model="m")

    def test_registers_system_prompts_from_defaults(self, tmp_path):
        enh = PromptEnhancer(
            providers=[_StaticProvider("p", "ok")],
            model="m",
        )
        # Defaults live inside _DEFAULT_SYSTEM_PROMPTS; no hot-reload file
        # means the enhancer falls back to the shipped values.
        assert "prompt enhancer" in enh._system_prompts.enhance.lower()

    def test_reads_override_files_when_present(self, tmp_path):
        (tmp_path / "enhance.txt").write_text("customized enhance")
        (tmp_path / "auto_extend.txt").write_text("")  # empty -> default
        enh = PromptEnhancer(
            providers=[_StaticProvider("p", "ok")],
            model="m",
            system_prompt_dir=str(tmp_path),
        )
        assert enh._system_prompts.enhance == "customized enhance"
        # Empty file fell back to the default.
        assert "continuation assistant" in enh._system_prompts.auto_extend


class TestEnhance:

    def test_enhance_returns_primary_provider_response(self):
        enh = PromptEnhancer(
            providers=[_StaticProvider("primary", "enhanced!")],
            model="m",
        )
        response = asyncio.run(enh.enhance("a fox"))
        assert response.content == "enhanced!"
        assert response.provider == "primary"
        assert response.fallback_used is False

    def test_enhance_falls_back_on_provider_error(self):
        enh = PromptEnhancer(
            providers=[
                _FailingProvider("a"),
                _StaticProvider("b", "fallback!"),
            ],
            model="m",
        )
        response = asyncio.run(enh.enhance("x"))
        assert response.provider == "b"
        assert response.content == "fallback!"
        assert response.fallback_used is True

    def test_enhance_stops_on_non_retryable_error(self):
        enh = PromptEnhancer(
            providers=[
                _FailingProvider("a", message="hard-fail", retryable=False),
                _StaticProvider("b", "should-not-be-reached"),
            ],
            model="m",
        )
        with pytest.raises(LLMProviderError, match="hard-fail"):
            asyncio.run(enh.enhance("x"))

    def test_enhance_raises_when_all_providers_fail(self):
        enh = PromptEnhancer(
            providers=[
                _FailingProvider("a"),
                _FailingProvider("b"),
            ],
            model="m",
        )
        with pytest.raises(LLMProviderError):
            asyncio.run(enh.enhance("x"))


class TestAutoExtendAndRewrite:

    def test_auto_extend_joins_prior_prompts(self):
        captured: list[str] = []

        class _Capturer:
            name = "cap"

            async def complete(self, request: LLMRequest) -> LLMResponse:
                user = next(m.content for m in request.messages
                            if m.role == "user")
                captured.append(user)
                return LLMResponse(
                    content="next",
                    provider=self.name,
                    model=request.model,
                    latency_ms=1.0,
                )

        enh = PromptEnhancer(providers=[_Capturer()], model="m")
        asyncio.run(enh.auto_extend(["first", "second"]))
        assert captured[0] == "first\nsecond"

    def test_rewrite_passes_seed_through(self):
        captured: list[str] = []

        class _Capturer:
            name = "cap"

            async def complete(self, request: LLMRequest) -> LLMResponse:
                user = next(m.content for m in request.messages
                            if m.role == "user")
                captured.append(user)
                return LLMResponse(
                    content="one\ntwo\nthree",
                    provider=self.name,
                    model=request.model,
                    latency_ms=1.0,
                )

        enh = PromptEnhancer(providers=[_Capturer()], model="m")
        response = asyncio.run(enh.rewrite("seed prompt"))
        assert captured == ["seed prompt"]
        assert response.content.splitlines() == ["one", "two", "three"]


class TestRegisterProvider:

    def test_register_appends_by_default(self):
        enh = PromptEnhancer(
            providers=[_StaticProvider("primary", "a")],
            model="m",
        )
        enh.register_provider(_StaticProvider("extra", "b"))
        assert [p.name for p in enh.providers] == ["primary", "extra"]

    def test_register_priority_zero_makes_primary(self):
        enh = PromptEnhancer(
            providers=[_StaticProvider("old", "a")],
            model="m",
        )
        enh.register_provider(
            _StaticProvider("new-primary", "b"), priority=0)
        assert enh.providers[0].name == "new-primary"

    def test_registered_provider_is_used_in_enhance(self):
        enh = PromptEnhancer(
            providers=[_FailingProvider("broken")],
            model="m",
        )
        enh.register_provider(_StaticProvider("fallback", "ok"))
        response = asyncio.run(enh.enhance("x"))
        assert response.content == "ok"
        assert response.fallback_used is True


class TestHotReload:

    def test_reload_picks_up_new_file(self, tmp_path):
        (tmp_path / "enhance.txt").write_text("first version")
        enh = PromptEnhancer(
            providers=[_StaticProvider("p", "ok")],
            model="m",
            system_prompt_dir=str(tmp_path),
        )
        assert enh._system_prompts.enhance == "first version"
        (tmp_path / "enhance.txt").write_text("second version")
        enh.reload_system_prompts()
        assert enh._system_prompts.enhance == "second version"
