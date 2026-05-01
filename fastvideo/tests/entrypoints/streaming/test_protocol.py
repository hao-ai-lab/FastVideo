# SPDX-License-Identifier: Apache-2.0
"""Protocol schema tests for the streaming server.

Covers:

* accepted client messages parse into the correct discriminated model
* unknown ``type`` values raise validation errors
* server-side messages serialize to the expected wire shape
* continuation_state field on session_init_v2 carries through
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fastvideo.entrypoints.streaming.protocol import (
    ContinuationStateSnapshot,
    ErrorMessage,
    GpuAssigned,
    Ltx2SegmentComplete,
    Ltx2SegmentStart,
    Ltx2StreamStart,
    MediaInit,
    MediaSegmentComplete,
    QueueStatus,
    SegmentPromptSource,
    SessionInitV2,
    SnapshotState,
    StepComplete,
    parse_client_message,
)


class TestClientMessageParsing:

    def test_session_init_v2_minimal(self):
        parsed = parse_client_message({"type": "session_init_v2"})
        assert isinstance(parsed, SessionInitV2)
        assert parsed.curated_prompts == []
        assert parsed.stream_mode == "av_fmp4"

    def test_session_init_v2_full(self):
        raw = {
            "type": "session_init_v2",
            "client_id": "client-1",
            "preset": "ltx2_two_stage",
            "preset_label": "2x refine",
            "curated_prompts": ["a fox", "a deer"],
            "enhancement_enabled": True,
            "auto_extension_enabled": False,
            "loop_generation_enabled": False,
            "single_clip_mode": True,
            "stream_mode": "av_fmp4",
            "continuation_state": {
                "kind": "ltx2.v1",
                "payload": {"schema_version": 1, "segment_index": 2},
            },
        }
        parsed = parse_client_message(raw)
        assert isinstance(parsed, SessionInitV2)
        assert parsed.preset == "ltx2_two_stage"
        assert parsed.curated_prompts == ["a fox", "a deer"]
        assert parsed.continuation_state["kind"] == "ltx2.v1"

    def test_segment_prompt_source(self):
        parsed = parse_client_message({
            "type": "segment_prompt_source",
            "prompt": "hello world",
            "source": "curated",
            "seed": 7,
        })
        assert isinstance(parsed, SegmentPromptSource)
        assert parsed.source == "curated"
        assert parsed.seed == 7

    def test_snapshot_state(self):
        parsed = parse_client_message({"type": "snapshot_state"})
        assert isinstance(parsed, SnapshotState)

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            parse_client_message({"type": "not_a_real_message"})

    def test_missing_type_rejected(self):
        with pytest.raises(ValidationError):
            parse_client_message({"prompt": "x"})

    def test_segment_prompt_source_requires_prompt(self):
        with pytest.raises(ValidationError):
            parse_client_message({"type": "segment_prompt_source"})


class TestServerMessageSerialization:

    def test_queue_status(self):
        msg = QueueStatus(position=3, queue_depth=5)
        assert msg.model_dump() == {
            "type": "queue_status",
            "position": 3,
            "queue_depth": 5,
        }

    def test_gpu_assigned(self):
        msg = GpuAssigned(gpu_id=1, session_timeout=300)
        assert msg.model_dump()["type"] == "gpu_assigned"

    def test_ltx2_stream_start(self):
        msg = Ltx2StreamStart(
            preset="ltx2_two_stage",
            width=1024, height=1536, fps=24, num_frames=121,
        )
        dumped = msg.model_dump()
        assert dumped["type"] == "ltx2_stream_start"
        assert dumped["width"] == 1024

    def test_ltx2_segment_start(self):
        msg = Ltx2SegmentStart(
            segment_idx=0,
            prompt="a fox",
            total_steps=8,
        )
        assert msg.model_dump()["segment_idx"] == 0

    def test_step_complete(self):
        msg = StepComplete(segment_idx=0, step=1, total_steps=8)
        assert msg.model_dump()["stage"] == "denoise"

    def test_media_init_has_mode(self):
        msg = MediaInit(segment_idx=0, stream_id="abc")
        dumped = msg.model_dump()
        assert dumped["mode"] == "av_fmp4"
        assert "avc1" in dumped["mime"]

    def test_media_segment_complete(self):
        msg = MediaSegmentComplete(
            segment_idx=0, stream_id="abc", chunks=4,
        )
        dumped = msg.model_dump()
        assert dumped["chunks"] == 4

    def test_ltx2_segment_complete(self):
        msg = Ltx2SegmentComplete(segment_idx=0, generation_time_ms=1234.5)
        assert msg.model_dump()["generation_time_ms"] == 1234.5

    def test_error_message_code_restricted(self):
        with pytest.raises(ValidationError):
            ErrorMessage(code="not_a_code", message="x")

    def test_continuation_state_snapshot(self):
        msg = ContinuationStateSnapshot(state={
            "kind": "ltx2.v1",
            "payload": {"schema_version": 1},
        })
        assert msg.model_dump()["state"]["kind"] == "ltx2.v1"
