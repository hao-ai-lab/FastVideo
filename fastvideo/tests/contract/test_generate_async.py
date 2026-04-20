# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ``VideoGenerator.generate_async``.

These tests monkey-patch the synchronous ``_generate_request_impl`` so
the suite runs CPU-only -- the async wrapper is the piece under test,
not the pipeline.
"""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from fastvideo.api.results import (
    GenerationResult,
    VideoFinalEvent,
    VideoProgressEvent,
)
from fastvideo.api.schema import (
    ContinuationState,
    GenerationRequest,
    OutputConfig,
    SamplingConfig,
)


class _FakeExecutor:

    def set_log_queue(self, q):
        self.log_queue = q

    def clear_log_queue(self):
        self.log_queue = None


class _FakeVideoGenerator:
    """Stand-in that exposes the same generate_async contract as the
    real VideoGenerator without requiring a loaded pipeline. Binds the
    real method via ``__func__`` so the implementation under test is
    exactly the production code."""

    def __init__(self, result: GenerationResult | list[GenerationResult]):
        self._result = result
        self.executor = _FakeExecutor()

    def _generate_request_impl(
        self, request: GenerationRequest,
    ) -> GenerationResult | list[GenerationResult]:
        return self._result

    @staticmethod
    def _wrap_legacy_result(result):
        return GenerationResult.from_legacy_result(result)

    @classmethod
    def bind(cls, result):
        from fastvideo.entrypoints.video_generator import VideoGenerator

        instance = cls(result)
        instance.generate_async = VideoGenerator.generate_async.__get__(
            instance, cls)
        instance.default_health_check_request = (
            VideoGenerator.default_health_check_request)
        return instance


def _make_request(**overrides: Any) -> GenerationRequest:
    base = GenerationRequest(
        prompt="hi",
        sampling=SamplingConfig(num_inference_steps=4, height=64, width=64,
                                 num_frames=8),
        output=OutputConfig(save_video=False, return_frames=False),
    )
    for key, value in overrides.items():
        setattr(base.sampling, key, value)
    return base


def _make_result(state: ContinuationState | None = None) -> GenerationResult:
    return GenerationResult(
        prompt="hi",
        frames=[np.zeros((64, 64, 3), dtype=np.uint8)],
        samples=None,
        generation_time=0.01,
        state=state,
    )


class TestEventOrdering:

    def test_emits_progress_then_final(self):
        gen = _FakeVideoGenerator.bind(_make_result())

        async def run():
            events = []
            async for evt in gen.generate_async(_make_request()):
                events.append(evt)
            return events

        events = asyncio.run(run())
        assert len(events) == 2
        assert isinstance(events[0], VideoProgressEvent)
        assert isinstance(events[1], VideoFinalEvent)
        assert events[0].step == 0
        assert events[0].total_steps == 4

    def test_exactly_one_final_event_per_request(self):
        gen = _FakeVideoGenerator.bind(_make_result())

        async def run():
            events = []
            async for evt in gen.generate_async(_make_request()):
                events.append(evt)
            return events

        events = asyncio.run(run())
        finals = [e for e in events if isinstance(e, VideoFinalEvent)]
        assert len(finals) == 1

    def test_batch_expansion_emits_one_final_per_result(self):
        batch_result = [_make_result(), _make_result()]
        gen = _FakeVideoGenerator.bind(batch_result)

        async def run():
            events = []
            async for evt in gen.generate_async(_make_request()):
                events.append(evt)
            return events

        events = asyncio.run(run())
        finals = [e for e in events if isinstance(e, VideoFinalEvent)]
        assert len(finals) == 2


class TestFinalEventShape:

    def test_carries_frames_and_result(self):
        result = _make_result()
        gen = _FakeVideoGenerator.bind(result)

        async def run() -> VideoFinalEvent:
            async for evt in gen.generate_async(_make_request()):
                if isinstance(evt, VideoFinalEvent):
                    return evt
            raise AssertionError("no final event")

        final = asyncio.run(run())
        assert final.frames is result.frames
        assert final.result is result
        assert final.metadata["generation_time"] == 0.01

    def test_carries_continuation_state_when_present(self):
        state = ContinuationState(
            kind="ltx2.v1", payload={"schema_version": 1, "segment_index": 3})
        result = _make_result(state=state)
        gen = _FakeVideoGenerator.bind(result)

        async def run() -> VideoFinalEvent:
            async for evt in gen.generate_async(_make_request()):
                if isinstance(evt, VideoFinalEvent):
                    return evt
            raise AssertionError("no final")

        final = asyncio.run(run())
        assert final.continuation_state is state

    def test_no_state_when_result_has_none(self):
        gen = _FakeVideoGenerator.bind(_make_result())

        async def run() -> VideoFinalEvent:
            async for evt in gen.generate_async(_make_request()):
                if isinstance(evt, VideoFinalEvent):
                    return evt
            raise AssertionError("no final")

        final = asyncio.run(run())
        assert final.continuation_state is None


class TestHealthCheckRequest:

    def test_returns_minimal_workload(self):
        from fastvideo.entrypoints.video_generator import VideoGenerator

        req = VideoGenerator.default_health_check_request()
        assert isinstance(req, GenerationRequest)
        assert req.sampling.num_inference_steps == 1
        assert req.sampling.num_frames == 8
        assert req.sampling.width == 256
        assert req.sampling.height == 256

    def test_does_not_persist_output(self):
        from fastvideo.entrypoints.video_generator import VideoGenerator

        req = VideoGenerator.default_health_check_request()
        assert req.output.save_video is False
        assert req.output.return_frames is False

    def test_round_trips_through_normalization(self):
        from fastvideo.api.compat import normalize_generation_request
        from fastvideo.entrypoints.video_generator import VideoGenerator

        req = VideoGenerator.default_health_check_request()
        normalized = normalize_generation_request(req)
        assert normalized.prompt == "health check"
        assert normalized.sampling.num_inference_steps == 1


class TestPublicExports:

    def test_new_event_types_available_via_fastvideo_api(self):
        import fastvideo.api as api

        for name in (
                "VideoEvent",
                "VideoFinalEvent",
                "VideoPartialEvent",
                "VideoProgressEvent",
                "VideoResult",
        ):
            assert name in api.__all__

    def test_video_result_is_generation_result_alias(self):
        from fastvideo.api import GenerationResult as Public
        from fastvideo.api import VideoResult

        assert VideoResult is Public


class TestDynamoStyleHandlerIntegration:
    """Mirror the shape the Dynamo backend package uses."""

    def test_async_handler_yields_typed_events_without_internal_imports(self):
        # The import set is exactly what the Dynamo backend uses.
        from fastvideo.api import (  # noqa: F401
            ContinuationState,
            GenerationRequest,
            InputConfig,
            OutputConfig,
            SamplingConfig,
            VideoFinalEvent,
            VideoProgressEvent,
        )

        gen = _FakeVideoGenerator.bind(_make_result())

        async def handler(request_dict, context):
            # Adapter: dict -> typed request.
            req = GenerationRequest(
                prompt=request_dict["prompt"],
                sampling=SamplingConfig(
                    num_inference_steps=1, num_frames=8,
                    height=256, width=256),
                output=OutputConfig(save_video=False, return_frames=False),
            )
            async for event in gen.generate_async(req):
                if isinstance(event, VideoFinalEvent):
                    yield {"type": "final"}
                elif isinstance(event, VideoProgressEvent):
                    yield {"type": "progress", "step": event.step}

        async def drive():
            out = []
            async for chunk in handler({"prompt": "x"}, None):
                out.append(chunk)
            return out

        chunks = asyncio.run(drive())
        types = [c["type"] for c in chunks]
        assert "progress" in types
        assert types[-1] == "final"
