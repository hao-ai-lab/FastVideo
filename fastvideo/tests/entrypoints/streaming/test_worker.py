# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :mod:`fastvideo.entrypoints.streaming.worker`.

The full ``worker_main`` loop runs in a subprocess and is exercised via
``test_gpu_pool.py``'s subprocess integration tests. This file covers
the in-process pieces:

* the two-segment warmup feeds segment 1's continuation state into
  segment 2 so both compile branches are primed before the worker
  reports ready
* result-shape extractors handle both attribute-style and dict-style
  generator returns
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any

from fastvideo.api.schema import (
    ContinuationState,
    GeneratorConfig,
    GenerationRequest,
    WarmupConfig,
)
from fastvideo.entrypoints.streaming.worker import (
    _extract_continuation_state,
    _warmup_worker,
    worker_main,
)


@dataclass
class _RecordingGenerator:
    """Captures the requests passed to ``generate`` and returns a
    canned :class:`ContinuationState` on the first call so the warmup
    can feed it into the second call.
    """

    state_to_return: ContinuationState | None = field(default_factory=lambda: ContinuationState(
        kind="ltx2.v1",
        payload={
            "schema_version": 1,
            "segment_index": 1
        },
    ))
    requests: list[GenerationRequest] = field(default_factory=list)

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        self.requests.append(request)
        return {"frames": [], "state": self.state_to_return}


class TestWarmupTwoSegment:

    def test_warmup_runs_segment_one_then_segment_two_with_returned_state(self) -> None:
        gen = _RecordingGenerator()
        _warmup_worker(gen, WarmupConfig(enabled=True, prompt="warm"))

        assert len(gen.requests) == 2

        seg1 = gen.requests[0]
        assert seg1.state is None
        assert seg1.output.return_state is True

        seg2 = gen.requests[1]
        assert seg2.state is gen.state_to_return

    def test_warmup_passes_through_when_no_state_returned(self) -> None:
        gen = _RecordingGenerator(state_to_return=None)
        _warmup_worker(gen, WarmupConfig(enabled=True, prompt="warm"))

        assert len(gen.requests) == 2
        assert gen.requests[1].state is None


class TestExtractContinuationState:

    def test_extracts_from_attribute(self) -> None:

        class _R:
            state = ContinuationState(kind="k", payload={})

        assert _extract_continuation_state(_R()).kind == "k"

    def test_extracts_from_dict(self) -> None:
        state = ContinuationState(kind="k", payload={})
        assert _extract_continuation_state({"state": state}) is state

    def test_returns_none_for_missing(self) -> None:
        assert _extract_continuation_state({}) is None
        assert _extract_continuation_state(object()) is None


def test_worker_rejects_external_launcher_before_generator_construction() -> None:
    result_queue: queue.Queue = queue.Queue()
    config = GeneratorConfig(model_path="/models/fake")
    config.engine.execution_backend = "external_launcher"

    worker_main(
        gpu_id=0,
        worker_id="worker-0",
        generator_config=config,
        warmup_config=WarmupConfig(enabled=False),
        job_queue=queue.Queue(),
        result_queue=result_queue,
        shutdown_event=threading.Event(),
    )

    result = result_queue.get_nowait()
    assert result["kind"] == "error"
    assert "external_launcher is supported only" in result["error"]
    assert "streaming worker" in result["error"]
