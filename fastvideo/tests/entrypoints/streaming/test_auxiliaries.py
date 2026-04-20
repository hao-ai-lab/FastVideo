# SPDX-License-Identifier: Apache-2.0
"""Tests for PR 7.8 streaming auxiliaries.

Covers:

* PromptSafetyFilter gracefully disables when fastText isn't installed
* SafetyResult semantics (allow, block, unavailable)
* RewriteOptions + _split_response parsing behavior
* SessionLogger JSONL append semantics + close lifecycle
* MockServer builds an app that drives the WS protocol end-to-end
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from fastvideo.entrypoints.streaming.prompt.safety import (
    PromptSafetyFilter,
    SafetyDecision,
    first_blocked,
)
from fastvideo.entrypoints.streaming.prompt.rewrite import (
    RewriteOptions,
    _split_response,
    build_rewrite,
)
from fastvideo.entrypoints.streaming.session_logger import (
    SessionLogEvent,
    SessionLogger,
)


# ----------------------------------------------------------------------
# Safety
# ----------------------------------------------------------------------


class TestPromptSafetyFilter:

    def test_disabled_by_default_when_no_path(self):
        f = PromptSafetyFilter(classifier_path=None)
        assert f.enabled is False
        result = f.classify("hi")
        assert result.decision is SafetyDecision.UNAVAILABLE

    def test_disabled_when_enabled_false(self):
        f = PromptSafetyFilter(classifier_path="/tmp/m.bin", enabled=False)
        assert f.enabled is False

    def test_unavailable_when_fasttext_missing(self, monkeypatch):
        # Force `import fasttext` inside _ensure_loaded to fail.
        monkeypatch.setitem(sys.modules, "fasttext", None)
        f = PromptSafetyFilter(classifier_path="/tmp/m.bin", enabled=True)
        result = f.classify("hi")
        assert result.decision is SafetyDecision.UNAVAILABLE

    def test_block_when_classifier_flags_unsafe(self, monkeypatch, tmp_path):
        fake_model = types.SimpleNamespace(
            predict=lambda text, k=1: (["__label__unsafe"], [0.95]))
        stub = types.SimpleNamespace(load_model=lambda _p: fake_model)
        monkeypatch.setitem(sys.modules, "fasttext", stub)
        model_path = str(tmp_path / "m.bin")
        Path(model_path).write_text("")
        f = PromptSafetyFilter(classifier_path=model_path, enabled=True)
        result = f.classify("please")
        assert result.decision is SafetyDecision.BLOCK
        assert result.label == "unsafe"
        assert result.score == pytest.approx(0.95)

    def test_allow_when_classifier_flags_safe(self, monkeypatch, tmp_path):
        fake_model = types.SimpleNamespace(
            predict=lambda text, k=1: (["__label__safe"], [0.99]))
        stub = types.SimpleNamespace(load_model=lambda _p: fake_model)
        monkeypatch.setitem(sys.modules, "fasttext", stub)
        f = PromptSafetyFilter(classifier_path="ignored", enabled=True)
        result = f.classify("hello")
        assert result.decision is SafetyDecision.ALLOW

    def test_below_threshold_allows_even_if_unsafe_label(self, monkeypatch):
        fake_model = types.SimpleNamespace(
            predict=lambda text, k=1: (["__label__unsafe"], [0.3]))
        stub = types.SimpleNamespace(load_model=lambda _p: fake_model)
        monkeypatch.setitem(sys.modules, "fasttext", stub)
        f = PromptSafetyFilter(
            classifier_path="m", enabled=True, block_threshold=0.5)
        assert f.classify("x").decision is SafetyDecision.ALLOW

    def test_first_blocked_returns_first_hit(self, monkeypatch):
        responses = iter([
            (["__label__safe"], [0.9]),
            (["__label__unsafe"], [0.9]),
            (["__label__safe"], [0.9]),
        ])
        fake_model = types.SimpleNamespace(
            predict=lambda text, k=1: next(responses))
        stub = types.SimpleNamespace(load_model=lambda _p: fake_model)
        monkeypatch.setitem(sys.modules, "fasttext", stub)
        f = PromptSafetyFilter(classifier_path="m", enabled=True)
        blocked = first_blocked(f, ["ok", "bad", "also ok"])
        assert blocked is not None
        assert blocked.prompt == "bad"


# ----------------------------------------------------------------------
# Rewrite
# ----------------------------------------------------------------------


class TestRewriteSplit:

    def test_plain_lines(self):
        assert _split_response("one\ntwo\nthree", limit=3) == [
            "one", "two", "three",
        ]

    def test_numbered_list(self):
        assert _split_response("1. first\n2. second", limit=3) == [
            "first", "second",
        ]

    def test_bulleted_list(self):
        assert _split_response("- one\n* two\n• three", limit=3) == [
            "one", "two", "three",
        ]

    def test_respects_limit(self):
        assert _split_response("a\nb\nc\nd", limit=2) == ["a", "b"]

    def test_limit_min_one(self):
        assert _split_response("only one", limit=0) == ["only one"]


class _StubEnhancer:

    async def rewrite(self, seed):
        from fastvideo.entrypoints.streaming.prompt.providers.base import LLMResponse

        return LLMResponse(
            content="1. alpha\n2. beta\n3. gamma",
            provider="stub",
            model="m",
            latency_ms=1.0,
        )


class TestBuildRewrite:

    def test_empty_seed_rejected(self):
        import asyncio

        with pytest.raises(ValueError):
            asyncio.run(build_rewrite(_StubEnhancer(), "   "))

    def test_returns_limited_alternatives(self):
        import asyncio

        result = asyncio.run(build_rewrite(
            _StubEnhancer(), "seed",
            options=RewriteOptions(count=2)))
        assert result.seed_prompt == "seed"
        assert result.alternatives == ["alpha", "beta"]
        assert result.provider == "stub"


# ----------------------------------------------------------------------
# Session logger
# ----------------------------------------------------------------------


class TestSessionLogger:

    def test_no_log_dir_is_noop(self):
        logger = SessionLogger(None)
        logger.log(SessionLogEvent(session_id="s", event="x"))  # no raise

    def test_appends_jsonl(self, tmp_path):
        logger = SessionLogger(str(tmp_path))
        logger.log(SessionLogEvent(
            session_id="s1",
            event="start",
            payload={"preset": "ltx2"},
            ts=1.0,
        ))
        logger.log(SessionLogEvent(
            session_id="s1",
            event="segment",
            payload={"idx": 0},
            ts=2.0,
        ))
        logger.close("s1")
        path = tmp_path / "session-s1.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["event"] == "start"
        assert first["payload"]["preset"] == "ltx2"

    def test_separate_files_per_session(self, tmp_path):
        logger = SessionLogger(str(tmp_path))
        logger.log(SessionLogEvent(session_id="a", event="e"))
        logger.log(SessionLogEvent(session_id="b", event="e"))
        logger.close_all()
        assert (tmp_path / "session-a.jsonl").exists()
        assert (tmp_path / "session-b.jsonl").exists()


# ----------------------------------------------------------------------
# Mock server
# ----------------------------------------------------------------------


class TestMockServer:

    def test_build_mock_app_returns_fastapi(self):
        from fastapi import FastAPI

        from fastvideo.entrypoints.streaming.mock_server import build_mock_app

        app = build_mock_app()
        assert isinstance(app, FastAPI)

    def test_mock_generator_produces_frames(self):
        from fastvideo.api.schema import GenerationRequest, SamplingConfig
        from fastvideo.entrypoints.streaming.mock_server import MockGenerator

        gen = MockGenerator()
        result = gen.generate(GenerationRequest(
            prompt="x",
            sampling=SamplingConfig(
                num_frames=3, height=32, width=32, num_inference_steps=1),
        ))
        assert len(result["frames"]) == 3
        assert result["frames"][0].shape == (32, 32, 3)
        assert result["state"].kind == "ltx2.v1"

    def test_mock_app_health_endpoint(self):
        from starlette.testclient import TestClient

        from fastvideo.entrypoints.streaming.mock_server import build_mock_app

        app = build_mock_app()
        client = TestClient(app)
        assert client.get("/health").json()["status"] == "ok"

    def test_mock_app_ws_handshake(self):
        from starlette.testclient import TestClient

        from fastvideo.entrypoints.streaming.mock_server import build_mock_app

        app = build_mock_app()
        client = TestClient(app)
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "session_init_v2"})
            assert ws.receive_json()["type"] == "queue_status"
            assert ws.receive_json()["type"] == "gpu_assigned"
            assert ws.receive_json()["type"] == "ltx2_stream_start"
