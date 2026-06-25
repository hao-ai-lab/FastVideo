# pyright: reportArgumentType=false
from __future__ import annotations

import asyncio
import os

from fastapi import WebSocketDisconnect


os.environ.setdefault("CEREBRAS_API_KEY", "dummy")
os.environ.setdefault("GROQ_API_KEY", "dummy")

import dreamverse.mock_server as mock_server


class _FakeWebSocket:
    def __init__(self, messages: list[tuple[float, dict[str, object]]]):
        self._messages = messages
        self._index = 0
        self.sent_json: list[dict[str, object]] = []
        self.sent_bytes: list[bytes] = []

    async def accept(self) -> None:
        return None

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent_json.append(payload)

    async def send_bytes(self, payload: bytes) -> None:
        self.sent_bytes.append(payload)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        del code, reason

    async def receive_json(self) -> dict[str, object]:
        if self._index >= len(self._messages):
            raise WebSocketDisconnect()
        delay_s, payload = self._messages[self._index]
        self._index += 1
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        return payload


def test_mock_server_matches_current_single5s_protocol():
    old_segment_bytes = mock_server.MOCK_SEGMENT_BYTES
    old_latency_ms = mock_server.LATENCY_MS
    try:
        mock_server.MOCK_SEGMENT_BYTES = b"mock-fmp4-bytes"
        mock_server.LATENCY_MS = 1

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "simple_prompt_1",
                        "curated_prompts": ["selected prompt"],
                        "single_clip_mode": True,
                        "enhancement_enabled": False,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (
                    0.01,
                    {
                        "type": "simple_generate",
                        "preset_id": "simple_custom_prompt",
                        "prompt_id": "simple_custom_prompt",
                        "prompt": "custom prompt",
                        "enhancement_enabled": True,
                        "initial_image": None,
                    },
                ),
                (0.20, {"type": "leave"}),
            ]
        )

        asyncio.run(mock_server.websocket_endpoint(ws))

        message_types = [payload["type"] for payload in ws.sent_json]
        assert message_types[0] == "queue_status"
        assert "gpu_assigned" in message_types
        assert message_types.count("ltx2_stream_start") == 2
        assert "prompt_received" in message_types
        assert "prompt_enhancing" in message_types
        assert "prompt_ready" in message_types
        assert "media_init" in message_types
        assert "media_segment_complete" in message_types
        assert message_types.count("ltx2_stream_complete") == 2
        assert "prompt_sources_blocked" not in message_types

        segment_start_events = [
            payload
            for payload in ws.sent_json
            if payload["type"] == "ltx2_segment_start"
        ]
        gpu_assigned_event = next(
            payload for payload in ws.sent_json if payload["type"] == "gpu_assigned"
        )
        assert gpu_assigned_event["session_timeout"] == mock_server.SESSION_TIMEOUT_SECONDS
        assert [payload["segment_idx"] for payload in segment_start_events] == [1, 1]
        assert segment_start_events[0]["prompt"] == "selected prompt"
        assert segment_start_events[1]["prompt"] == "custom prompt"

        step_complete_events = [
            payload
            for payload in ws.sent_json
            if payload["type"] == "step_complete"
        ]
        assert len(step_complete_events) == 2
        assert step_complete_events[0]["latency_ms"] == {
            "total": 121.0,
            "worker_e2e": 1.0,
            "main_user_step": 121.0,
            "overhead": 120.0,
        }

        assert ws.sent_bytes
        assert ws.sent_bytes[0] == b"mock-fmp4-bytes"
    finally:
        mock_server.MOCK_SEGMENT_BYTES = old_segment_bytes
        mock_server.LATENCY_MS = old_latency_ms


def test_mock_server_regular_cap_waits_for_rewrite_rollout():
    old_segment_bytes = mock_server.MOCK_SEGMENT_BYTES
    old_latency_ms = mock_server.LATENCY_MS
    old_generation_segment_cap = mock_server.GENERATION_SEGMENT_CAP
    try:
        mock_server.MOCK_SEGMENT_BYTES = b"mock-fmp4-bytes"
        mock_server.LATENCY_MS = 1
        mock_server.GENERATION_SEGMENT_CAP = 1

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "test_preset",
                        "curated_prompts": ["segment one"],
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (
                    0.02,
                    {
                        "type": "rewrite_seed_prompts",
                        "rewrite_instruction": "start a new rollout",
                    },
                ),
                (0.20, {"type": "leave"}),
            ]
        )

        asyncio.run(mock_server.websocket_endpoint(ws))

        message_types = [payload["type"] for payload in ws.sent_json]
        assert message_types.count("ltx2_stream_start") == 2
        assert message_types.count("ltx2_stream_complete") == 2
        assert "generation_cap_reached" not in message_types
        assert "prompt_sources_blocked" not in message_types

        segment_start_events = [
            payload
            for payload in ws.sent_json
            if payload["type"] == "ltx2_segment_start"
        ]
        assert [payload["segment_idx"] for payload in segment_start_events] == [1, 1]
        assert segment_start_events[0]["prompt"] == "segment one"
        assert segment_start_events[1]["prompt"] == "segment one [start a new rollout]"
    finally:
        mock_server.MOCK_SEGMENT_BYTES = old_segment_bytes
        mock_server.LATENCY_MS = old_latency_ms
        mock_server.GENERATION_SEGMENT_CAP = old_generation_segment_cap


def test_mock_server_rewrite_during_active_segment_restarts_from_first_rewritten_prompt():
    old_segment_bytes = mock_server.MOCK_SEGMENT_BYTES
    old_latency_ms = mock_server.LATENCY_MS
    try:
        mock_server.MOCK_SEGMENT_BYTES = b"mock-fmp4-bytes"
        mock_server.LATENCY_MS = 100

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "test_preset",
                        "curated_prompts": ["segment one", "segment two"],
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (
                    0.02,
                    {
                        "type": "rewrite_seed_prompts",
                        "rewrite_instruction": "restart from rewrite",
                    },
                ),
                (0.40, {"type": "leave"}),
            ]
        )

        asyncio.run(mock_server.websocket_endpoint(ws))

        segment_start_events = [
            payload
            for payload in ws.sent_json
            if payload["type"] == "ltx2_segment_start"
        ]
        assert [payload["prompt"] for payload in segment_start_events[:2]] == [
            "segment one",
            "segment one [restart from rewrite]",
        ]
        assert all(
            payload["prompt"] != "segment two"
            for payload in segment_start_events[1:]
        )
        reset_events = [
            payload
            for payload in ws.sent_json
            if payload.get("type") == "seed_prompts_reset_applied"
        ]
        assert any(
            payload.get("reason") == "rewrite_during_generation"
            for payload in reset_events
        )
    finally:
        mock_server.MOCK_SEGMENT_BYTES = old_segment_bytes
        mock_server.LATENCY_MS = old_latency_ms


def test_mock_server_supports_initial_custom_rollout_prompt():
    old_segment_bytes = mock_server.MOCK_SEGMENT_BYTES
    old_latency_ms = mock_server.LATENCY_MS
    try:
        mock_server.MOCK_SEGMENT_BYTES = b"mock-fmp4-bytes"
        mock_server.LATENCY_MS = 1

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "custom_editable",
                        "preset_label": "Custom rollout",
                        "curated_prompts": [],
                        "initial_rollout_prompt": "A moonbase corridor thriller with flooding",
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (0.20, {"type": "leave"}),
            ]
        )

        asyncio.run(mock_server.websocket_endpoint(ws))

        message_types = [payload["type"] for payload in ws.sent_json]
        assert "rewrite_seed_prompts_started" in message_types
        assert "seed_prompts_updated" in message_types
        assert "rewrite_seed_prompts_complete" in message_types
        assert "seed_prompts_reset_applied" in message_types
        assert "ltx2_stream_start" in message_types
        assert "prompt_sources_blocked" not in message_types

        segment_start_events = [
            payload
            for payload in ws.sent_json
            if payload["type"] == "ltx2_segment_start"
        ]
        assert segment_start_events
        assert segment_start_events[0]["prompt"] == (
            "A moonbase corridor thriller with flooding [segment 1]"
        )
    finally:
        mock_server.MOCK_SEGMENT_BYTES = old_segment_bytes
        mock_server.LATENCY_MS = old_latency_ms


def test_mock_server_can_start_new_project_without_reconnecting():
    old_segment_bytes = mock_server.MOCK_SEGMENT_BYTES
    old_latency_ms = mock_server.LATENCY_MS
    try:
        mock_server.MOCK_SEGMENT_BYTES = b"mock-fmp4-bytes"
        mock_server.LATENCY_MS = 40

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "test_preset",
                        "curated_prompts": ["segment one"],
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (0.02, {"type": "end_project_keep_session"}),
                (
                    0.20,
                    {
                        "type": "project_init_v1",
                        "preset_id": "test_preset_2",
                        "preset_label": "Test Preset 2",
                        "curated_prompts": ["segment two"],
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (0.40, {"type": "leave"}),
            ]
        )

        asyncio.run(mock_server.websocket_endpoint(ws))

        message_types = [payload["type"] for payload in ws.sent_json]
        assert message_types.count("gpu_assigned") == 1
        assert message_types.count("ltx2_stream_start") == 2
        assert "project_idle" in message_types

        project_idle_index = message_types.index("project_idle")
        stream_start_indexes = [
            index for index, message_type in enumerate(message_types)
            if message_type == "ltx2_stream_start"
        ]
        assert stream_start_indexes[0] < project_idle_index < stream_start_indexes[1]

        segment_start_events = [
            payload
            for payload in ws.sent_json
            if payload["type"] == "ltx2_segment_start"
        ]
        assert [payload["prompt"] for payload in segment_start_events[:2]] == [
            "segment one",
            "segment two",
        ]
    finally:
        mock_server.MOCK_SEGMENT_BYTES = old_segment_bytes
        mock_server.LATENCY_MS = old_latency_ms
