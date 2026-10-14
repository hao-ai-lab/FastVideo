# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for how `fastvideo bench` measures request latency."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from fastvideo.entrypoints.cli import bench_serving
from fastvideo.entrypoints.cli.bench_serving import (
    RequestFuncInput,
    RequestFuncOutput,
    calculate_metrics,
    limited_request,
)


def _output(latency: float, queue_wait: float, success: bool = True) -> RequestFuncOutput:
    return RequestFuncOutput(success=success, latency=latency, queue_wait=queue_wait)


def test_e2e_latency_includes_the_wait_for_a_concurrency_slot():
    # Five requests due at once, served one at a time in 1 s each: the server
    # sees 1 s per request, the k-th user waits k seconds.
    outputs = [_output(latency=1.0, queue_wait=float(k)) for k in range(5)]
    outputs.append(_output(latency=0.1, queue_wait=9.0, success=False))

    metrics = calculate_metrics(outputs, total_duration=5.0)

    assert metrics["latency_p99"] == pytest.approx(1.0)
    assert metrics["e2e_latency_mean"] == pytest.approx(3.0)
    assert metrics["e2e_latency_median"] == pytest.approx(3.0)
    assert metrics["e2e_latency_p99"] == pytest.approx(np.percentile([1, 2, 3, 4, 5], 99))
    assert metrics["failed_requests"] == 1


def test_e2e_latency_equals_latency_without_a_wait():
    outputs = [_output(latency=x, queue_wait=0.0) for x in (0.5, 1.5, 2.5)]

    metrics = calculate_metrics(outputs, total_duration=3.0)

    for stat in ("mean", "median", "p99"):
        assert metrics[f"e2e_latency_{stat}"] == pytest.approx(metrics[f"latency_{stat}"])


def test_limited_request_records_the_wait_for_a_slot(monkeypatch):
    # A fake clock that moves only when the test moves it, so the wait is exact.
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(bench_serving, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    started = {"a": asyncio.Event(), "b": asyncio.Event()}
    done = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def fake_request(req, session, pbar):
        output = RequestFuncOutput(start_time=clock.now)
        started[req.prompt].set()
        await done[req.prompt].wait()
        output.latency = clock.now - output.start_time
        output.success = True
        return output

    async def run() -> list:
        semaphore = asyncio.Semaphore(1)
        reqs = [RequestFuncInput(prompt=p, api_url="http://unused", model="m") for p in ("a", "b")]
        # Both are due at t=0; "a" takes the only slot and is served until t=3.
        tasks = [asyncio.create_task(limited_request(fake_request, r, None, None, semaphore)) for r in reqs]
        await started["a"].wait()
        clock.now = 3.0
        done["a"].set()
        await started["b"].wait()
        clock.now = 5.0
        done["b"].set()
        return await asyncio.gather(*tasks)

    a, b = asyncio.run(run())

    assert (a.queue_wait, a.latency) == (0.0, 3.0)
    assert (b.queue_wait, b.latency) == (3.0, 2.0)
