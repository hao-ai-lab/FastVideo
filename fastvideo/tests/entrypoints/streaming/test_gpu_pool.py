# SPDX-License-Identifier: Apache-2.0
"""GPU pool tests.

InProcessGpuPool is exercised end-to-end. SubprocessGpuPool is driven
with an injected ``worker_factory`` that stands up a fake worker inside
a thread (not a subprocess) so the test suite stays CPU-only.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest

from fastvideo.api.schema import (
    GeneratorConfig,
    GenerationRequest,
    GpuPoolConfig,
    WarmupConfig,
)
from fastvideo.entrypoints.streaming.gpu_pool import (
    GpuPool,
    InProcessGpuPool,
    PoolAcquireTimeout,
    SubprocessGpuPool,
    _WorkerHandle,
)


# ----------------------------------------------------------------------
# In-process pool
# ----------------------------------------------------------------------


@dataclass
class _MockGenerator:
    sleep_s: float = 0.0

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return {
            "frames": [],
            "prompt_echo": request.prompt,
        }


class TestInProcessGpuPool:

    def test_is_gpu_pool(self):
        assert isinstance(
            InProcessGpuPool(_MockGenerator()), GpuPool)

    def test_acquire_returns_deterministic_assignment(self):
        pool = InProcessGpuPool(_MockGenerator(), gpu_id=7)

        async def run():
            a = await pool.acquire("sess-a")
            return a

        a = asyncio.run(run())
        assert a.gpu_id == 7
        assert a.worker_id.startswith("inproc-")

    def test_acquire_is_sticky_across_calls(self):
        pool = InProcessGpuPool(_MockGenerator())

        async def run():
            a = await pool.acquire("sess-a")
            b = await pool.acquire("sess-a")
            return a, b

        a, b = asyncio.run(run())
        assert a == b

    def test_run_without_acquire_raises(self):
        pool = InProcessGpuPool(_MockGenerator())

        async def run():
            with pytest.raises(RuntimeError):
                await pool.run(
                    "sess-a", GenerationRequest(prompt="hi"))

        asyncio.run(run())

    def test_run_returns_generator_output(self):
        pool = InProcessGpuPool(_MockGenerator())

        async def run():
            await pool.acquire("sess-a")
            return await pool.run(
                "sess-a", GenerationRequest(prompt="hi"))

        result = asyncio.run(run())
        assert result["prompt_echo"] == "hi"

    def test_release_frees_binding(self):
        pool = InProcessGpuPool(_MockGenerator())

        async def run():
            await pool.acquire("sess-a")
            await pool.release("sess-a")
            with pytest.raises(RuntimeError):
                await pool.run(
                    "sess-a", GenerationRequest(prompt="hi"))

        asyncio.run(run())

    def test_health_reports_active_sessions(self):
        pool = InProcessGpuPool(_MockGenerator())

        async def run():
            await pool.acquire("sess-a")
            health = pool.health()
            assert health.total_workers == 1
            assert health.active_sessions == 1
            assert health.available_workers == 0

        asyncio.run(run())


# ----------------------------------------------------------------------
# Subprocess pool (driven by a thread-backed fake worker factory)
# ----------------------------------------------------------------------


class _ThreadWorker:
    """Stand-in for a subprocess worker.

    Runs a Python thread that pulls jobs from ``job_queue`` and invokes
    a supplied mock generator. The control flow matches
    :func:`worker_main` exactly (ready + result dict shapes) so the
    parent-side pool under test exercises the same code paths.
    """

    def __init__(
        self,
        generator: _MockGenerator,
        *,
        job_queue: mp.Queue,
        result_queue: mp.Queue,
        shutdown_event: threading.Event,
        warmup_ms: float = 0.0,
    ) -> None:
        self._generator = generator
        self._job_queue = job_queue
        self._result_queue = result_queue
        self._shutdown_event = shutdown_event
        self._warmup_ms = warmup_ms
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        if self._warmup_ms:
            time.sleep(self._warmup_ms / 1000.0)
        self._result_queue.put({"kind": "ready"})
        while not self._shutdown_event.is_set():
            try:
                item = self._job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                result = self._generator.generate(item["request"])
                self._result_queue.put({
                    "kind": "result",
                    "job_id": item["job_id"],
                    "result": result,
                })
            except Exception as exc:  # pragma: no cover - defensive
                self._result_queue.put({
                    "kind": "error",
                    "job_id": item["job_id"],
                    "error": repr(exc),
                })


def _thread_worker_factory(generator_builder):
    """Return a WorkerFactory that uses thread workers instead of procs."""

    def factory(
        *,
        gpu_id: int,
        generator_config: GeneratorConfig,
        warmup_config: WarmupConfig,
    ) -> _WorkerHandle:
        ctx = mp.get_context("spawn")
        job_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()
        shutdown_event = threading.Event()
        ready = threading.Event()
        mp_shutdown = ctx.Event()

        generator = generator_builder(gpu_id)
        worker = _ThreadWorker(
            generator,
            job_queue=job_queue,
            result_queue=result_queue,
            shutdown_event=shutdown_event,
        )
        worker.start()

        # Drain the ready sentinel from the queue in the same way the
        # real factory does (thread waiter populates ``ready``).
        def _await_ready() -> None:
            while True:
                try:
                    msg = result_queue.get(timeout=1.0)
                except queue.Empty:
                    if shutdown_event.is_set():
                        return
                    continue
                if msg.get("kind") == "ready":
                    ready.set()
                    return

        threading.Thread(target=_await_ready, daemon=True).start()

        class _FakeProcess:
            def __init__(self, stop: threading.Event, worker: _ThreadWorker):
                self._stop = stop
                self._worker = worker

            def is_alive(self) -> bool:
                return self._worker._thread.is_alive()

            def join(self, timeout: float | None = None) -> None:
                self._stop.set()
                self._worker.join(timeout)

            def kill(self) -> None:
                self._stop.set()

        fake_process = _FakeProcess(shutdown_event, worker)

        return _WorkerHandle(
            process=fake_process,  # type: ignore[arg-type]
            job_queue=job_queue,
            result_queue=result_queue,
            gpu_id=gpu_id,
            worker_id=f"gpu{gpu_id}-fake",
            ready=ready,
            shutdown_event=mp_shutdown,
        )

    return factory


@pytest.fixture
def pool_factory():
    """Provide a SubprocessGpuPool built against thread workers."""

    async def _build(num_workers: int = 2):
        pool = SubprocessGpuPool(
            generator_config=GeneratorConfig(model_path="/models/fake"),
            pool_config=GpuPoolConfig(num_workers=num_workers),
            warmup_config=WarmupConfig(enabled=False),
            worker_factory=_thread_worker_factory(
                lambda gpu_id: _MockGenerator()),
        )
        await pool.start()
        return pool

    return _build


class TestSubprocessGpuPool:

    def test_start_spawns_requested_workers(self, pool_factory):
        async def run():
            pool = await pool_factory(num_workers=3)
            try:
                h = pool.health()
                assert h.total_workers == 3
                assert h.available_workers == 3
            finally:
                await pool.shutdown()

        asyncio.run(run())

    def test_acquire_decrements_available(self, pool_factory):
        async def run():
            pool = await pool_factory(num_workers=2)
            try:
                a = await pool.acquire("sess-a")
                assert a.worker_id.endswith("-fake")
                assert pool.health().available_workers == 1
            finally:
                await pool.shutdown()

        asyncio.run(run())

    def test_acquire_timeout_when_all_busy(self, pool_factory):
        async def run():
            pool = await pool_factory(num_workers=1)
            try:
                await pool.acquire("sess-a")
                with pytest.raises(PoolAcquireTimeout):
                    await pool.acquire("sess-b", timeout=0.1)
            finally:
                await pool.shutdown()

        asyncio.run(run())

    def test_run_returns_worker_result(self, pool_factory):
        async def run():
            pool = await pool_factory(num_workers=1)
            try:
                await pool.acquire("sess-a")
                result = await pool.run(
                    "sess-a", GenerationRequest(prompt="hello"))
                assert result["prompt_echo"] == "hello"
            finally:
                await pool.shutdown()

        asyncio.run(run())

    def test_release_returns_worker_to_pool(self, pool_factory):
        async def run():
            pool = await pool_factory(num_workers=1)
            try:
                await pool.acquire("sess-a")
                await pool.release("sess-a")
                # Now a second acquire should succeed without timeout.
                a = await pool.acquire("sess-b", timeout=1.0)
                assert a.worker_id.endswith("-fake")
            finally:
                await pool.shutdown()

        asyncio.run(run())

    def test_sticky_binding_across_multiple_runs(self, pool_factory):
        async def run():
            pool = await pool_factory(num_workers=2)
            try:
                a1 = await pool.acquire("sess-a")
                a2 = await pool.acquire("sess-a")
                assert a1.worker_id == a2.worker_id
                # Two runs land on the same worker.
                await pool.run("sess-a", GenerationRequest(prompt="1"))
                await pool.run("sess-a", GenerationRequest(prompt="2"))
            finally:
                await pool.shutdown()

        asyncio.run(run())

    def test_run_without_acquire_raises(self, pool_factory):
        async def run():
            pool = await pool_factory(num_workers=1)
            try:
                with pytest.raises(RuntimeError):
                    await pool.run(
                        "sess-x", GenerationRequest(prompt="x"))
            finally:
                await pool.shutdown()

        asyncio.run(run())

    def test_shutdown_is_idempotent(self, pool_factory):
        async def run():
            pool = await pool_factory(num_workers=2)
            await pool.shutdown()
            await pool.shutdown()  # no raise

        asyncio.run(run())
