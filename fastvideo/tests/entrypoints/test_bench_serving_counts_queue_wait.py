# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for how `fastvideo bench` measures request latency."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from fastvideo.entrypoints.cli.bench_serving import (
    RequestFuncInput,
    RequestFuncOutput,
    calculate_metrics,
    limited_request,
)


def _output(latency: float, queue_wait: float, success: bool = True) -> RequestFuncOutput:
    return RequestFuncOutput(success=success, latency=latency, queue_wait=queue_wait)


def test_e2e_latency_includes_the_wait_for_a_concurrency_slot():
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


def test_limited_request_counts_the_wait_from_arrival():
    clock = SimpleNamespace(now=0.0)
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
        tasks = [asyncio.create_task(limited_request(fake_request, r, None, None, semaphore, 0.0)) for r in reqs]
        clock.now = 0.5
        await started["a"].wait()
        clock.now = 3.0
        done["a"].set()
        await started["b"].wait()
        clock.now = 5.0
        done["b"].set()
        return await asyncio.gather(*tasks)

    a, b = asyncio.run(run())

    assert (a.queue_wait, a.latency) == (0.5, 2.5)
    assert (b.queue_wait, b.latency) == (3.0, 2.0)
