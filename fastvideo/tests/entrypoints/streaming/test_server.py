# SPDX-License-Identifier: Apache-2.0
"""End-to-end WebSocket smoke for the streaming server skeleton.

Uses a mock generator so these tests run CPU-only (no GPU, no model
weights). Skips the fMP4 assertions when ``ffmpeg`` is missing.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

from fastvideo.api.schema import (  # noqa: E402
    ContinuationState,
    GeneratorConfig,
    SamplingConfig,
    ServeConfig,
    StreamingConfig,
    GenerationRequest,
)
from fastvideo.entrypoints.streaming.server import build_app  # noqa: E402


_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@dataclass
class _MockGenerator:
    width: int = 64
    height: int = 64
    fps: int = 12
    num_frames: int = 12
    return_state: bool = True

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        frames = [
            np.full((self.height, self.width, 3), i * 5, dtype=np.uint8)
            for i in range(self.num_frames)
        ]
        state = (ContinuationState(
            kind="ltx2.v1",
            payload={
                "schema_version": 1,
                "segment_index": 0,
                "source_prompt": request.prompt,
            },
        ) if self.return_state else None)
        return {
            "frames": frames,
            "audio_sample_rate": 24000,
            "state": state,
        }


def _build_serve_config() -> ServeConfig:
    return ServeConfig(
        generator=GeneratorConfig(model_path="/models/fake"),
        default_request=GenerationRequest(
            sampling=SamplingConfig(
                num_frames=12,
                height=64,
                width=64,
                fps=12,
                num_inference_steps=1,
            ),
        ),
        streaming=StreamingConfig(
            session_timeout_seconds=60,
            generation_segment_cap=2,
        ),
    )


def _build_client() -> tuple[TestClient, _MockGenerator]:
    generator = _MockGenerator()
    app = build_app(_build_serve_config(), generator)
    return TestClient(app), generator


class TestHealth:

    def test_health_endpoint_reports_stream_mode(self):
        client, _ = _build_client()
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["stream_mode"] == "av_fmp4"
        assert body["sessions"] == 0


class TestSessionHandshake:

    def test_rejects_non_session_init_opening_frame(self):
        client, _ = _build_client()
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "segment_prompt_source", "prompt": "x"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "invalid_message"

    def test_rejects_unknown_message_on_init(self):
        client, _ = _build_client()
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "not_a_message"})
            err = ws.receive_json()
            assert err["type"] == "error"

    def test_emits_queue_and_gpu_assigned_on_valid_init(self):
        client, _ = _build_client()
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({
                "type": "session_init_v2",
                "preset": "ltx2_two_stage",
                "curated_prompts": ["a fox"],
            })
            assert ws.receive_json()["type"] == "queue_status"
            assert ws.receive_json()["type"] == "gpu_assigned"
            assert ws.receive_json()["type"] == "ltx2_stream_start"

    def test_init_hydrates_continuation_state(self):
        client, _ = _build_client()
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({
                "type": "session_init_v2",
                "preset": "ltx2_two_stage",
                "continuation_state": {
                    "kind": "ltx2.v1",
                    "payload": {"schema_version": 1, "segment_index": 3},
                },
            })
            # Drain handshake frames
            ws.receive_json()  # queue_status
            ws.receive_json()  # gpu_assigned
            ws.receive_json()  # ltx2_stream_start
            # Ask the server for the state back; it should echo what we sent.
            ws.send_json({"type": "snapshot_state"})
            snap = ws.receive_json()
            assert snap["type"] == "continuation_state_snapshot"
            assert snap["state"]["kind"] == "ltx2.v1"
            assert snap["state"]["payload"]["segment_index"] == 3


@pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg not installed")
class TestSegmentFlow:

    def test_segment_generates_media_init_plus_complete(self):
        client, generator = _build_client()
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "session_init_v2",
                          "preset": "ltx2_two_stage"})
            for _ in range(3):
                ws.receive_json()  # queue_status + gpu_assigned + stream_start

            ws.send_json({
                "type": "segment_prompt_source",
                "prompt": "a test segment",
                "num_inference_steps": 1,
            })
            start = ws.receive_json()
            assert start["type"] == "ltx2_segment_start"
            assert start["segment_idx"] == 0
            step = ws.receive_json()
            assert step["type"] == "step_complete"
            media_init = ws.receive_json()
            assert media_init["type"] == "media_init"
            # Then one or more binary frames until media_segment_complete.
            saw_binary = False
            while True:
                msg = ws.receive()
                if "bytes" in msg and msg["bytes"]:
                    saw_binary = True
                    continue
                parsed = _as_json(msg)
                if parsed is None:
                    continue
                if parsed["type"] == "media_segment_complete":
                    break
            assert saw_binary
            final = ws.receive_json()
            assert final["type"] == "ltx2_segment_complete"
            assert final["segment_idx"] == 0


class TestContinuationStatePersistence:

    def test_snapshot_after_segment_carries_generator_state(self):
        if not _FFMPEG_AVAILABLE:
            pytest.skip("ffmpeg not installed")
        client, generator = _build_client()
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "session_init_v2",
                          "preset": "ltx2_two_stage"})
            for _ in range(3):
                ws.receive_json()
            ws.send_json({
                "type": "segment_prompt_source",
                "prompt": "a cat",
                "num_inference_steps": 1,
            })
            _drain_until(ws, "ltx2_segment_complete")
            ws.send_json({"type": "snapshot_state"})
            snap = ws.receive_json()
            assert snap["type"] == "continuation_state_snapshot"
            assert snap["state"]["kind"] == "ltx2.v1"
            assert snap["state"]["payload"]["source_prompt"] == "a cat"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _drain_until(ws, target_type: str) -> dict[str, Any]:
    while True:
        msg = ws.receive()
        if "text" in msg and msg["text"]:
            import json

            parsed = json.loads(msg["text"])
            if parsed.get("type") == target_type:
                return parsed
        # skip binary / other


def _as_json(msg: dict[str, Any]) -> dict[str, Any] | None:
    if "text" not in msg or not msg["text"]:
        return None
    import json

    return json.loads(msg["text"])
