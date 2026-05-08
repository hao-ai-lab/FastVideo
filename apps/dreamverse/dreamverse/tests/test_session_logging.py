from __future__ import annotations
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportMissingTypeArgument=false, reportOperatorIssue=false

import asyncio
import base64
import io
import json
import os
import re
import socket
from types import SimpleNamespace

from fastapi import WebSocketDisconnect
from PIL import Image
import pytest


os.environ.setdefault("CEREBRAS_API_KEY", "dummy")
os.environ.setdefault("GROQ_API_KEY", "dummy")

import dreamverse.main as server_main  # noqa: E402
import dreamverse.runtime as runtime  # noqa: E402
from dreamverse.session_logger import SessionEventLogger  # noqa: E402
from dreamverse.worker_ipc import MediaChunk, MediaComplete, MediaInit  # noqa: E402
from dreamverse.session import controller as session_controller  # noqa: E402

pytestmark = pytest.mark.gpu


class _FakeSessionLogger:
    def __init__(self):
        self.events: list[dict[str, object]] = []

    async def write_event(
        self,
        *,
        event: str,
        client_id: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        entry: dict[str, object] = {
            "event": event,
            "client_id": client_id,
        }
        if payload:
            entry.update(payload)
        self.events.append(entry)


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


class _FakeSlot:
    def __init__(self, step_delay_s: float = 0.0):
        self.shared_stream_buffer = None
        self._stream_queues: dict[str, asyncio.Queue] = {}
        self.calls: list[dict[str, object]] = []
        self._step_delay_s = max(0.0, float(step_delay_s))

    async def join_user(self, client_id: str, model_id: str) -> None:
        del client_id, model_id

    def register_stream_queue(self, client_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._stream_queues[client_id] = queue
        return queue

    def unregister_stream_queue(self, client_id: str) -> None:
        self._stream_queues.pop(client_id, None)

    async def user_step(
        self,
        client_id: str,
        prompt: str,
        segment_idx: int,
        reset_conditioning: bool,
        image_path: str | None = None,
    ):
        self.calls.append({
            "client_id": client_id,
            "prompt": prompt,
            "segment_idx": segment_idx,
            "image_path": image_path,
            "reset_conditioning": reset_conditioning,
        })
        queue = self._stream_queues[client_id]
        await queue.put(MediaInit(
            user_id=client_id,
            segment_idx=segment_idx,
            stream_id=f"stream-{segment_idx}",
            mime='video/mp4; codecs="avc1.640028,mp4a.40.2"',
            uses_shared_buffer=False,
        ))
        await queue.put(MediaChunk(
            user_id=client_id,
            segment_idx=segment_idx,
            stream_id=f"stream-{segment_idx}",
            chunk=b"abc123",
            uses_shared_buffer=False,
        ))
        await queue.put(MediaComplete(
            user_id=client_id,
            segment_idx=segment_idx,
            stream_id=f"stream-{segment_idx}",
            chunks=1,
        ))
        if self._step_delay_s > 0:
            await asyncio.sleep(self._step_delay_s)
        return {"e2e_latency_ms": 10.0}


class _FakeGPUPool:
    def __init__(self, step_delay_s: float = 0.0):
        self._slot = _FakeSlot(step_delay_s=step_delay_s)

    def get_status(self) -> dict[str, int]:
        return {
            "queue_size": 0,
            "available_gpus": 1,
            "total_gpus": 1,
        }

    async def acquire(self, client_id: str, websocket) -> tuple[int, _FakeSlot]:
        del client_id, websocket
        return 0, self._slot

    async def release(self, client_id: str) -> None:
        del client_id


def _make_png_data_url() -> str:
    image = Image.new("RGB", (2, 2), color=(16, 32, 64))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class _FakePromptEnhancer:
    def set_provider(self, provider: object, preferred_model: object = None):
        del provider, preferred_model
        return {
            "provider": "openai",
            "provider_options": ["openai", "cerebras"],
            "options": ["gpt-5-nano-2025-08-07"],
        }

    def get_rewrite_model_config(self):
        return self.set_provider(None, None)

    def resolve_rewrite_model(self, value: object) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "gpt-5-nano-2025-08-07"

    def resolve_rewrite_system_prompt(self, value: object) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "shared rewrite system prompt"

    def resolve_rewrite_new_rollout_system_prompt(self, value: object) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "new rollout rewrite system prompt"

    def resolve_rewrite_temperature(self, value: float | None) -> float:
        return 1.0 if value is None else float(value)

    async def enhance_prompt(
        self,
        raw_prompt: str,
        *,
        locked_segments: list[str],
        next_segment_idx: int,
        preset_id: str | None,
        mode: str,
        model: str,
        timeout_ms: int,
    ):
        del locked_segments, next_segment_idx, preset_id, mode, model, timeout_ms
        return SimpleNamespace(
            prompt=f"enhanced: {raw_prompt}",
            fallback_used=False,
            error=None,
            provider="openai",
            model="gpt-5-nano-2025-08-07",
            latency_ms=12.5,
        )

    async def rewrite_prompt_sequence(
        self,
        snapshot_prompts: list[str],
        *,
        preset_id: str | None,
        preset_label: str | None,
        rewrite_instruction: str,
        rewrite_model: str,
        rewrite_temperature: float,
        timeout_ms: int,
        system_prompt_override: str | None = None,
    ):
        del (
            snapshot_prompts,
            preset_id,
            preset_label,
            rewrite_instruction,
            rewrite_model,
            rewrite_temperature,
            timeout_ms,
            system_prompt_override,
        )
        return SimpleNamespace(
            prompts=["rewrite a", "rewrite b"],
            fallback_used=False,
            error=None,
            provider="openai",
            model="gpt-5-nano-2025-08-07",
            latency_ms=23.0,
            rollout_id="rewrite_rollout",
            rollout_label="Rewrite Rollout",
            raw_response_text=(
                '{"id":"rewrite_rollout","label":"Rewrite Rollout",'
                '"segment_prompts":["rewrite a","rewrite b"]}'
            ),
        )


class _FallbackPromptEnhancer(_FakePromptEnhancer):
    async def enhance_prompt(
        self,
        raw_prompt: str,
        *,
        locked_segments: list[str],
        next_segment_idx: int,
        preset_id: str | None,
        mode: str,
        model: str,
        timeout_ms: int,
    ):
        del (
            raw_prompt,
            locked_segments,
            next_segment_idx,
            preset_id,
            mode,
            model,
            timeout_ms,
        )
        return SimpleNamespace(
            prompt="",
            fallback_used=True,
            error="mock enhancement timeout",
            provider="openai",
            model="gpt-5-nano-2025-08-07",
            latency_ms=18.0,
        )


class _UnsafeEnhancedPromptEnhancer(_FakePromptEnhancer):
    async def enhance_prompt(
        self,
        raw_prompt: str,
        *,
        locked_segments: list[str],
        next_segment_idx: int,
        preset_id: str | None,
        mode: str,
        model: str,
        timeout_ms: int,
    ):
        del (
            raw_prompt,
            locked_segments,
            next_segment_idx,
            preset_id,
            mode,
            model,
            timeout_ms,
        )
        return SimpleNamespace(
            prompt="unsafe enhanced prompt",
            fallback_used=False,
            error=None,
            provider="openai",
            model="gpt-5-nano-2025-08-07",
            latency_ms=14.0,
        )


class _FakePromptSafetyFilter:
    def __init__(self, blocked_prompts: dict[str, str]):
        self._blocked_prompts = {
            str(prompt).strip(): str(error)
            for prompt, error in blocked_prompts.items()
            if str(prompt).strip()
        }

    def get_prompt_safety_error(self, prompt: str) -> str | None:
        return self._blocked_prompts.get(str(prompt).strip())

    def get_first_blocked_prompt(self, prompts: list[str]):
        for index, prompt in enumerate(prompts):
            error = self.get_prompt_safety_error(prompt)
            if error is not None:
                return SimpleNamespace(index=index, prompt=prompt, error=error)
        return None


def test_session_event_logger_initializes_hostname_folder_and_utc_filename(
    tmp_path,
):
    logger = SessionEventLogger(tmp_path)
    assert logger.directory == tmp_path / socket.gethostname()
    assert logger.directory.is_dir()
    assert re.match(r"^\d{6}_\d{6}\.jsonl$", logger.path.name)
    assert logger.path.is_file()


def test_session_event_logger_appends_valid_jsonl(tmp_path):
    logger = SessionEventLogger(tmp_path)

    async def _write() -> None:
        await logger.write_event(
            event="append_prompt",
            client_id="client-123",
            payload={"prompt": "hello world"},
        )

    asyncio.run(_write())

    lines = logger.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "append_prompt"
    assert record["client_id"] == "client-123"
    assert record["hostname"] == socket.gethostname()
    assert isinstance(record["ts"], str)
    assert record["prompt"] == "hello world"


def test_websocket_flow_emits_required_session_events():
    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    try:
        fake_logger = _FakeSessionLogger()
        runtime.gpu_pool = _FakeGPUPool()
        runtime.prompt_enhancer = _FakePromptEnhancer()
        runtime.session_event_logger = fake_logger

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "curated_prompts": ["segment one prompt"],
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (0.01, {"type": "append_prompt", "prompt": "make it rainy"}),
                (
                    0.01,
                    {
                        "type": "rewrite_seed_prompts",
                        "rewrite_instruction": "more dramatic",
                    },
                ),
                (0.30, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        event_names = [event["event"] for event in fake_logger.events]
        assert "ws_session_start" in event_names
        assert "gpu_assigned" in event_names
        assert "append_prompt" in event_names
        assert "segment_start" in event_names
        assert "segment_complete" in event_names

        rewrite_events = [
            event
            for event in fake_logger.events
            if event["event"] == "rewrite_done"
        ]
        rewrite_kinds = {event.get("kind") for event in rewrite_events}
        assert "seed_rewrite" in rewrite_kinds
        assert "enhance_prompt" in rewrite_kinds

        segment_complete = next(
            event
            for event in fake_logger.events
            if event["event"] == "segment_complete"
        )
        latency = segment_complete.get("latency_ms")
        assert isinstance(latency, dict)
        assert set(latency.keys()) == {
            "total",
            "worker_e2e",
            "main_user_step",
            "overhead",
        }
        assert isinstance(segment_complete.get("data_size_bytes"), int)
        assert segment_complete["data_size_bytes"] > 0
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger


def test_simple_generate_reuses_session_and_resets_conditioning():
    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool()
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = _FakePromptEnhancer()
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = None

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "simple_prompt_1",
                        "curated_prompts": ["a selected prompt"],
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
                        "prompt": "a fresh custom prompt",
                        "enhancement_enabled": True,
                        "initial_image": {
                            "name": "frame.png",
                            "mime_type": "image/png",
                            "data_url": _make_png_data_url(),
                        },
                    },
                ),
                (0.50, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert [call["segment_idx"] for call in fake_pool._slot.calls] == [1, 1]
        assert fake_pool._slot.calls[0]["prompt"] == "a selected prompt"
        assert fake_pool._slot.calls[0]["reset_conditioning"] is False
        assert fake_pool._slot.calls[0]["image_path"] is None
        assert fake_pool._slot.calls[1]["prompt"] == "enhanced: a fresh custom prompt"
        assert fake_pool._slot.calls[1]["reset_conditioning"] is True
        assert isinstance(fake_pool._slot.calls[1]["image_path"], str)

        event_names = [event["event"] for event in fake_logger.events]
        assert "simple_generate" in event_names

        stream_start_events = [
            payload
            for payload in ws.sent_json
            if payload.get("type") == "ltx2_stream_start"
        ]
        assert len(stream_start_events) == 2
        assert all(
            payload.get("generation_segment_cap") == 0
            for payload in stream_start_events
        )
        stream_complete_events = [
            payload
            for payload in ws.sent_json
            if payload.get("type") == "ltx2_stream_complete"
        ]
        assert len(stream_complete_events) == 2
        assert "prompt_sources_blocked" not in [
            payload.get("type") for payload in ws.sent_json
        ]

        segment_start_events = [
            payload
            for payload in ws.sent_json
            if payload.get("type") == "ltx2_segment_start"
        ]
        assert [payload["segment_idx"] for payload in segment_start_events] == [1, 1]

        step_complete_events = [
            payload
            for payload in ws.sent_json
            if payload.get("type") == "step_complete"
        ]
        assert len(step_complete_events) == 2
        assert all(
            set(payload.get("latency_ms", {}).keys()) == {
                "total",
                "worker_e2e",
                "main_user_step",
                "overhead",
            }
            for payload in step_complete_events
        )
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter


def test_single5s_ignores_global_segment_cap():
    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    old_generation_segment_cap = session_controller.GENERATION_SEGMENT_CAP
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool()
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = _FakePromptEnhancer()
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = None
        session_controller.GENERATION_SEGMENT_CAP = 1

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "simple_prompt_1",
                        "curated_prompts": ["a selected prompt"],
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
                        "prompt": "a fresh custom prompt",
                        "enhancement_enabled": True,
                        "initial_image": None,
                    },
                ),
                (0.50, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert [call["segment_idx"] for call in fake_pool._slot.calls] == [1, 1]
        assert all(
            payload.get("generation_segment_cap") == 0
            for payload in ws.sent_json
            if payload.get("type") == "ltx2_stream_start"
        )
        assert "generation_cap_reached" not in [
            payload.get("type") for payload in ws.sent_json
        ]
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter
        session_controller.GENERATION_SEGMENT_CAP = old_generation_segment_cap


def test_regular_segment_cap_completes_stream_and_allows_rewrite_rollout_restart():
    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    old_generation_segment_cap = session_controller.GENERATION_SEGMENT_CAP
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool()
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = _FakePromptEnhancer()
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = None
        session_controller.GENERATION_SEGMENT_CAP = 1

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "test_preset",
                        "curated_prompts": ["segment one prompt"],
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (
                    0.20,
                    {
                        "type": "rewrite_seed_prompts",
                        "rewrite_instruction": "start a new rollout",
                    },
                ),
                (0.50, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert [call["segment_idx"] for call in fake_pool._slot.calls] == [1, 1]
        assert fake_pool._slot.calls[0]["prompt"] == "segment one prompt"
        assert fake_pool._slot.calls[1]["prompt"] == "rewrite a"
        assert fake_pool._slot.calls[1]["reset_conditioning"] is True

        message_types = [payload["type"] for payload in ws.sent_json]
        assert message_types.count("ltx2_stream_start") == 2
        assert message_types.count("ltx2_stream_complete") == 2
        assert "generation_cap_reached" not in message_types
        assert "prompt_sources_blocked" not in message_types
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter
        session_controller.GENERATION_SEGMENT_CAP = old_generation_segment_cap


def test_rewrite_during_active_segment_restarts_from_rewritten_seed_window():
    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool(step_delay_s=0.15)
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = _FakePromptEnhancer()
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = None

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "test_preset",
                        "curated_prompts": [
                            "segment one prompt",
                            "segment two prompt",
                        ],
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (
                    0.01,
                    {
                        "type": "rewrite_seed_prompts",
                        "rewrite_instruction": "restart from rewrite",
                    },
                ),
                (0.45, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert [call["prompt"] for call in fake_pool._slot.calls[:2]] == [
            "segment one prompt",
            "rewrite a",
        ]
        assert all(
            call["prompt"] != "segment two prompt"
            for call in fake_pool._slot.calls[1:]
        )
        assert fake_pool._slot.calls[1]["segment_idx"] == 1
        assert fake_pool._slot.calls[1]["reset_conditioning"] is True

        reset_events = [
            payload
            for payload in ws.sent_json
            if payload.get("type") == "seed_prompts_reset_applied"
        ]
        assert reset_events
        assert any(
            payload.get("reason") == "rewrite_during_generation"
            for payload in reset_events
        )
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter


def test_initial_custom_rollout_prompt_generates_seed_window_before_streaming():
    class _InitialRolloutPromptEnhancer(_FakePromptEnhancer):
        def __init__(self):
            self.rewrite_calls: list[dict[str, object]] = []

        async def rewrite_prompt_sequence(
            self,
            snapshot_prompts: list[str],
            *,
            preset_id: str | None,
            preset_label: str | None,
            rewrite_instruction: str,
            rewrite_model: str,
            rewrite_temperature: float,
            timeout_ms: int,
            system_prompt_override: str | None = None,
        ):
            self.rewrite_calls.append({
                "snapshot_prompts": list(snapshot_prompts),
                "preset_id": preset_id,
                "preset_label": preset_label,
                "rewrite_instruction": rewrite_instruction,
                "rewrite_model": rewrite_model,
                "rewrite_temperature": rewrite_temperature,
                "timeout_ms": timeout_ms,
                "system_prompt_override": system_prompt_override,
            })
            del rewrite_model, rewrite_temperature, timeout_ms, system_prompt_override
            return SimpleNamespace(
                prompts=[
                    "rewrite 1",
                    "rewrite 2",
                    "rewrite 3",
                    "rewrite 4",
                    "rewrite 5",
                    "rewrite 6",
                ],
                fallback_used=False,
                error=None,
                provider="openai",
                model="gpt-5-nano-2025-08-07",
                latency_ms=23.0,
                rollout_id=preset_id or "custom_editable",
                rollout_label=preset_label or "Custom rollout",
                raw_response_text=(
                    '{"id":"custom_editable","label":"Custom rollout",'
                    '"segment_prompts":["rewrite 1","rewrite 2","rewrite 3",'
                    '"rewrite 4","rewrite 5","rewrite 6"]}'
                ),
            )

    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool()
        fake_enhancer = _InitialRolloutPromptEnhancer()
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = fake_enhancer
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = None

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
                        "rewrite_model": "gpt-4.1-mini",
                        "rewrite_temperature": 0.4,
                        "rewrite_window_system_prompt": "Session-specific rewrite prompt",
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (0.20, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert len(fake_enhancer.rewrite_calls) == 1
        assert fake_enhancer.rewrite_calls[0] == {
            "snapshot_prompts": [],
            "preset_id": "custom_editable",
            "preset_label": "Custom rollout",
            "rewrite_instruction": "A moonbase corridor thriller with flooding",
            "rewrite_model": "gpt-4.1-mini",
            "rewrite_temperature": 0.4,
            "timeout_ms": server_main.PROMPT_TIMEOUT_MS,
            "system_prompt_override": "Session-specific rewrite prompt",
        }
        assert fake_pool._slot.calls
        assert fake_pool._slot.calls[0]["prompt"] == "rewrite 1"

        message_types = [payload["type"] for payload in ws.sent_json]
        assert "rewrite_seed_prompts_started" in message_types
        assert "seed_prompts_updated" in message_types
        assert "rewrite_seed_prompts_complete" in message_types
        assert "seed_prompts_reset_applied" in message_types
        assert "ltx2_stream_start" in message_types
        assert "prompt_sources_blocked" not in message_types
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter


def test_initial_custom_rollout_prompt_uses_dedicated_user_prompt_by_default():
    class _InitialRolloutPromptEnhancer(_FakePromptEnhancer):
        def __init__(self):
            self.rewrite_calls: list[dict[str, object]] = []

        async def rewrite_prompt_sequence(
            self,
            snapshot_prompts: list[str],
            *,
            preset_id: str | None,
            preset_label: str | None,
            rewrite_instruction: str,
            rewrite_model: str,
            rewrite_temperature: float,
            timeout_ms: int,
            system_prompt_override: str | None = None,
        ):
            self.rewrite_calls.append({
                "snapshot_prompts": list(snapshot_prompts),
                "preset_id": preset_id,
                "preset_label": preset_label,
                "rewrite_instruction": rewrite_instruction,
                "rewrite_model": rewrite_model,
                "rewrite_temperature": rewrite_temperature,
                "timeout_ms": timeout_ms,
                "system_prompt_override": system_prompt_override,
            })
            return SimpleNamespace(
                prompts=[
                    "rewrite 1",
                    "rewrite 2",
                    "rewrite 3",
                    "rewrite 4",
                    "rewrite 5",
                    "rewrite 6",
                ],
                fallback_used=False,
                error=None,
                provider="openai",
                model="gpt-5-nano-2025-08-07",
                latency_ms=23.0,
                rollout_id=preset_id or "custom_editable",
                rollout_label=preset_label or "Custom rollout",
                raw_response_text=(
                    '{"id":"custom_editable","label":"Custom rollout",'
                    '"segment_prompts":["rewrite 1","rewrite 2","rewrite 3",'
                    '"rewrite 4","rewrite 5","rewrite 6"]}'
                ),
            )

    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool()
        fake_enhancer = _InitialRolloutPromptEnhancer()
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = fake_enhancer
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = None

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
                        "rewrite_model": "gpt-4.1-mini",
                        "rewrite_temperature": 0.4,
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (0.20, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert len(fake_enhancer.rewrite_calls) == 1
        assert fake_enhancer.rewrite_calls[0]["snapshot_prompts"] == []
        assert (
            fake_enhancer.rewrite_calls[0]["system_prompt_override"]
            == "new rollout rewrite system prompt"
        )
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter


def test_initial_custom_rollout_prompt_prefers_session_specific_user_prompt():
    class _InitialRolloutPromptEnhancer(_FakePromptEnhancer):
        def __init__(self):
            self.rewrite_calls: list[dict[str, object]] = []

        async def rewrite_prompt_sequence(
            self,
            snapshot_prompts: list[str],
            *,
            preset_id: str | None,
            preset_label: str | None,
            rewrite_instruction: str,
            rewrite_model: str,
            rewrite_temperature: float,
            timeout_ms: int,
            system_prompt_override: str | None = None,
        ):
            self.rewrite_calls.append({
                "snapshot_prompts": list(snapshot_prompts),
                "preset_id": preset_id,
                "preset_label": preset_label,
                "rewrite_instruction": rewrite_instruction,
                "rewrite_model": rewrite_model,
                "rewrite_temperature": rewrite_temperature,
                "timeout_ms": timeout_ms,
                "system_prompt_override": system_prompt_override,
            })
            return SimpleNamespace(
                prompts=[
                    "rewrite 1",
                    "rewrite 2",
                    "rewrite 3",
                    "rewrite 4",
                    "rewrite 5",
                    "rewrite 6",
                ],
                fallback_used=False,
                error=None,
                provider="openai",
                model="gpt-5-nano-2025-08-07",
                latency_ms=23.0,
                rollout_id=preset_id or "custom_editable",
                rollout_label=preset_label or "Custom rollout",
                raw_response_text=(
                    '{"id":"custom_editable","label":"Custom rollout",'
                    '"segment_prompts":["rewrite 1","rewrite 2","rewrite 3",'
                    '"rewrite 4","rewrite 5","rewrite 6"]}'
                ),
            )

    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool()
        fake_enhancer = _InitialRolloutPromptEnhancer()
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = fake_enhancer
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = None

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
                        "rewrite_model": "gpt-4.1-mini",
                        "rewrite_temperature": 0.4,
                        "rewrite_window_system_prompt": "Window rewrite prompt",
                        "rewrite_user_system_prompt": "User rewrite prompt",
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (0.20, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert len(fake_enhancer.rewrite_calls) == 1
        assert fake_enhancer.rewrite_calls[0]["snapshot_prompts"] == []
        assert fake_enhancer.rewrite_calls[0]["system_prompt_override"] == "User rewrite prompt"
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter


def test_single5s_raw_prompt_block_prevents_generation():
    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool()
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = _FakePromptEnhancer()
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = _FakePromptSafetyFilter({
            "blocked raw prompt": "Test raw prompt blocked.",
        })

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "simple_custom_prompt",
                        "curated_prompts": [],
                        "single_clip_mode": True,
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (
                    0.01,
                    {
                        "type": "append_prompt",
                        "prompt_id": "simple_custom_prompt",
                        "prompt": "blocked raw prompt",
                    },
                ),
                (0.20, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert fake_pool._slot.calls == []
        assert {
            payload["type"] for payload in ws.sent_json
        }.isdisjoint({"ltx2_segment_start", "ltx2_segment_complete"})
        error_payloads = [
            payload for payload in ws.sent_json if payload.get("type") == "error"
        ]
        assert any(
            payload.get("message") == "Test raw prompt blocked."
            for payload in error_payloads
        )

        prompt_blocked_events = [
            event
            for event in fake_logger.events
            if event["event"] == "prompt_blocked"
        ]
        assert len(prompt_blocked_events) == 1
        assert prompt_blocked_events[0]["kind"] == "user_raw"
        assert prompt_blocked_events[0]["raw_prompt"] == "blocked raw prompt"
        assert all(event["event"] != "segment_start" for event in fake_logger.events)
        assert all(event["event"] != "rewrite_done" for event in fake_logger.events)
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter


def test_single5s_enhanced_prompt_block_prevents_generation():
    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool()
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = _UnsafeEnhancedPromptEnhancer()
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = _FakePromptSafetyFilter({
            "unsafe enhanced prompt": "Test enhanced prompt blocked.",
        })

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "simple_custom_prompt",
                        "curated_prompts": [],
                        "single_clip_mode": True,
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (
                    0.01,
                    {
                        "type": "append_prompt",
                        "prompt_id": "simple_custom_prompt",
                        "prompt": "safe raw prompt",
                    },
                ),
                (0.20, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert fake_pool._slot.calls == []
        assert {
            payload["type"] for payload in ws.sent_json
        }.isdisjoint({"ltx2_segment_start", "ltx2_segment_complete"})
        error_payloads = [
            payload for payload in ws.sent_json if payload.get("type") == "error"
        ]
        assert any(
            payload.get("message") == "Test enhanced prompt blocked."
            for payload in error_payloads
        )

        rewrite_events = [
            event
            for event in fake_logger.events
            if event["event"] == "rewrite_done"
        ]
        assert len(rewrite_events) == 1
        assert rewrite_events[0]["kind"] == "enhance_prompt"

        prompt_blocked_events = [
            event
            for event in fake_logger.events
            if event["event"] == "prompt_blocked"
        ]
        assert len(prompt_blocked_events) == 1
        assert prompt_blocked_events[0]["kind"] == "user_enhanced"
        assert prompt_blocked_events[0]["output_prompt"] == "unsafe enhanced prompt"
        assert all(event["event"] != "segment_start" for event in fake_logger.events)
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter


def test_single5s_enhancement_fallback_does_not_start_generation():
    old_gpu_pool = runtime.gpu_pool
    old_prompt_enhancer = runtime.prompt_enhancer
    old_session_logger = runtime.session_event_logger
    old_prompt_safety_filter = runtime.prompt_safety_filter
    try:
        fake_logger = _FakeSessionLogger()
        fake_pool = _FakeGPUPool()
        runtime.gpu_pool = fake_pool
        runtime.prompt_enhancer = _FallbackPromptEnhancer()
        runtime.session_event_logger = fake_logger
        runtime.prompt_safety_filter = None

        ws = _FakeWebSocket(
            [
                (
                    0.0,
                    {
                        "type": "session_init_v2",
                        "preset_id": "simple_custom_prompt",
                        "curated_prompts": [],
                        "single_clip_mode": True,
                        "enhancement_enabled": True,
                        "auto_extension_enabled": False,
                        "loop_generation_enabled": False,
                    },
                ),
                (
                    0.01,
                    {
                        "type": "append_prompt",
                        "prompt_id": "simple_custom_prompt",
                        "prompt": "prompt that times out",
                    },
                ),
                (0.20, {"type": "leave"}),
            ]
        )

        asyncio.run(server_main.websocket_endpoint(ws))

        assert fake_pool._slot.calls == []
        assert {
            payload["type"] for payload in ws.sent_json
        }.isdisjoint({"ltx2_segment_start", "ltx2_segment_complete"})

        fallback_payloads = [
            payload
            for payload in ws.sent_json
            if payload.get("type") == "prompt_fallback_used"
        ]
        assert len(fallback_payloads) == 1
        assert fallback_payloads[0]["prompt_id"] == "simple_custom_prompt"
        assert (
            fallback_payloads[0]["error"]
            == server_main.PROMPT_EXTENSION_FAILURE_USER_MESSAGE
        )
        assert fallback_payloads[0]["source"] == "user_enhancement_failed"

        rewrite_events = [
            event
            for event in fake_logger.events
            if event["event"] == "rewrite_done"
        ]
        assert len(rewrite_events) == 1
        assert rewrite_events[0]["kind"] == "enhance_prompt"
        assert all(event["event"] != "segment_start" for event in fake_logger.events)
        assert all(event["event"] != "prompt_blocked" for event in fake_logger.events)
    finally:
        runtime.gpu_pool = old_gpu_pool
        runtime.prompt_enhancer = old_prompt_enhancer
        runtime.session_event_logger = old_session_logger
        runtime.prompt_safety_filter = old_prompt_safety_filter
