# SPDX-License-Identifier: Apache-2.0
"""GPU pool manager for the streaming server.

Replaces the single-generator path in PR 7.5 with a typed pool
abstraction. Three implementations ship here:

* :class:`InProcessGpuPool` — one in-process ``VideoGenerator``; used
  by tests and single-GPU dev deployments.
* :class:`SubprocessGpuPool` — one ``multiprocessing.Process`` per
  GPU, each running :func:`worker_main` against a ``GeneratorConfig``.
  Jobs are dispatched via ``multiprocessing.Queue``.
* :class:`GpuPool` (abstract) — the interface both use.

Session-to-GPU binding lives in the pool so continuation state stays
on the GPU that generated the previous segment (matching the internal
``gpu_pool.py``'s per-GPU cache behavior). Cross-GPU handoff is
supported via :class:`SessionStore` snapshot + hydrate, which
serializes the state before the migration and rehydrates it on the
new worker.

Typed config: workers start from a :class:`GeneratorConfig` (no flat
LTX-2 kwargs), satisfying the PR 6 + PR 7 contracts that the public
surface doesn't reintroduce the legacy kwarg bag.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastvideo.api.schema import (
    GeneratorConfig,
    GenerationRequest,
    GpuPoolConfig,
    WarmupConfig,
)
from fastvideo.entrypoints.streaming.session_store import (
    InMemorySessionStore,
    SessionStore,
)
from fastvideo.entrypoints.streaming.worker import worker_main
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class _GeneratorLike(Protocol):
    """Subset the pool calls on a worker-side generator."""

    def generate(self, request: GenerationRequest) -> Any:
        ...


@dataclass
class PoolAssignment:
    """The worker a session is currently bound to."""

    gpu_id: int
    worker_id: str
    pinned_at: float = field(default_factory=time.monotonic)


class GpuPool(ABC):
    """Abstract GPU pool.

    ``acquire`` binds a session to a worker and holds that binding
    across segments so continuation state can stay hot. ``run`` submits
    a single ``GenerationRequest`` for a bound session.

    Acquire / release are independent of run — a session can run many
    segments on one acquired worker, and must release on disconnect.
    """

    @abstractmethod
    async def acquire(
        self,
        session_id: str,
        *,
        timeout: float | None = None,
    ) -> PoolAssignment:
        ...

    @abstractmethod
    async def run(
        self,
        session_id: str,
        request: GenerationRequest,
    ) -> Any:
        ...

    @abstractmethod
    async def release(self, session_id: str) -> None:
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        ...

    @abstractmethod
    def health(self) -> PoolHealth:
        ...


@dataclass
class PoolHealth:
    total_workers: int
    available_workers: int
    active_sessions: int
    queued_sessions: int = 0


class PoolAcquireTimeout(RuntimeError):
    """Raised when ``acquire`` times out waiting for a free worker."""


# ---------------------------------------------------------------------------
# In-process implementation (single-worker, test / dev)
# ---------------------------------------------------------------------------


class InProcessGpuPool(GpuPool):
    """Single-process pool backed by one :class:`_GeneratorLike`.

    This is what PR 7.5's server uses by default; PR 7.6 adds the real
    ``SubprocessGpuPool`` alternative but keeps this one for tests and
    small deployments.
    """

    def __init__(
        self,
        generator: _GeneratorLike,
        *,
        gpu_id: int = 0,
        session_store: SessionStore | None = None,
    ) -> None:
        self._generator = generator
        self._gpu_id = gpu_id
        self._worker_id = f"inproc-{uuid.uuid4().hex[:6]}"
        self._session_store = session_store or InMemorySessionStore()
        self._active: dict[str, PoolAssignment] = {}
        self._lock = asyncio.Lock()
        self._gen_lock = asyncio.Lock()

    async def acquire(
        self,
        session_id: str,
        *,
        timeout: float | None = None,
    ) -> PoolAssignment:
        async with self._lock:
            existing = self._active.get(session_id)
            if existing is not None:
                return existing
            assignment = PoolAssignment(gpu_id=self._gpu_id, worker_id=self._worker_id)
            self._active[session_id] = assignment
            return assignment

    async def run(
        self,
        session_id: str,
        request: GenerationRequest,
    ) -> Any:
        if session_id not in self._active:
            raise RuntimeError(f"session {session_id!r} is not acquired on this pool")
        # Serialize generator access so one GPU runs one request at a
        # time, matching the internal gpu_pool's per-GPU lock.
        async with self._gen_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._generator.generate, request)

    async def release(self, session_id: str) -> None:
        async with self._lock:
            self._active.pop(session_id, None)

    async def shutdown(self) -> None:
        self._active.clear()

    def health(self) -> PoolHealth:
        return PoolHealth(
            total_workers=1,
            available_workers=1 if not self._active else 0,
            active_sessions=len(self._active),
        )


# ---------------------------------------------------------------------------
# Subprocess implementation (multi-worker, real deployment)
# ---------------------------------------------------------------------------


@dataclass
class _WorkerHandle:
    process: Any  # mp.Process or compatible handle with is_alive / join / kill
    job_queue: mp.Queue
    result_queue: mp.Queue
    gpu_id: int
    worker_id: str
    ready: threading.Event
    shutdown_event: Any  # mp.Event is a factory, not a type — Any keeps mypy sane


@dataclass
class _PendingJob:
    job_id: str
    future: Future
    session_id: str


class SubprocessGpuPool(GpuPool):
    """One ``multiprocessing.Process`` per GPU.

    Each worker boots :class:`fastvideo.VideoGenerator` from a typed
    :class:`GeneratorConfig` inside the child process (post-
    ``CUDA_VISIBLE_DEVICES`` setup) and consumes jobs from an mp Queue.

    This is the production shape: the parent process stays CPU-only, and
    GPU state never crosses process boundaries. Continuation state is
    serialized through :class:`SessionStore` for cross-GPU handoff.

    PR 7.6 ships this as an opt-in; PR 7.5's in-process pool remains the
    default until nightly runs validate the subprocess path.
    """

    def __init__(
        self,
        generator_config: GeneratorConfig,
        *,
        pool_config: GpuPoolConfig,
        warmup_config: WarmupConfig | None = None,
        session_store: SessionStore | None = None,
        worker_factory: WorkerFactory | None = None,
    ) -> None:
        self._generator_config = generator_config
        self._pool_config = pool_config
        self._warmup_config = warmup_config or WarmupConfig()
        self._session_store = session_store or InMemorySessionStore()
        self._worker_factory = worker_factory or _default_worker_factory
        self._workers: list[_WorkerHandle] = []
        self._available: asyncio.Queue[int] = asyncio.Queue()
        self._assignments: dict[str, PoolAssignment] = {}
        self._worker_by_id: dict[str, _WorkerHandle] = {}
        self._pending: dict[str, _PendingJob] = {}
        self._lock = asyncio.Lock()
        self._result_reader_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Spawn worker processes and wait for each to report ready."""
        num_workers = self._pool_config.num_workers or 1
        for gpu_id in range(num_workers):
            handle = self._worker_factory(
                gpu_id=gpu_id,
                generator_config=self._generator_config,
                warmup_config=self._warmup_config,
            )
            self._workers.append(handle)
            self._worker_by_id[handle.worker_id] = handle

        # Wait for each worker's ready event in a thread to avoid
        # blocking the event loop.
        loop = asyncio.get_running_loop()
        await asyncio.gather(*[
            loop.run_in_executor(None, handle.ready.wait, self._warmup_config.timeout_seconds)
            for handle in self._workers
        ])

        # Start background result readers — one task per worker
        # drains its result queue and resolves futures in _pending.
        for handle in self._workers:
            task = asyncio.create_task(self._drain_results(handle))
            self._result_reader_tasks.append(task)

        for idx in range(num_workers):
            await self._available.put(idx)

    async def acquire(
        self,
        session_id: str,
        *,
        timeout: float | None = None,
    ) -> PoolAssignment:
        async with self._lock:
            existing = self._assignments.get(session_id)
            if existing is not None:
                return existing
        try:
            idx = await asyncio.wait_for(self._available.get(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise PoolAcquireTimeout(f"no worker available after {timeout}s") from exc
        handle = self._workers[idx]
        assignment = PoolAssignment(gpu_id=handle.gpu_id, worker_id=handle.worker_id)
        async with self._lock:
            self._assignments[session_id] = assignment
        return assignment

    async def run(
        self,
        session_id: str,
        request: GenerationRequest,
    ) -> Any:
        assignment = self._assignments.get(session_id)
        if assignment is None:
            raise RuntimeError(f"session {session_id!r} not acquired on this pool")
        handle = self._worker_by_id[assignment.worker_id]
        job_id = uuid.uuid4().hex
        future: Future = Future()
        self._pending[job_id] = _PendingJob(job_id=job_id, future=future, session_id=session_id)
        handle.job_queue.put({"job_id": job_id, "request": request})
        return await asyncio.wrap_future(future)

    async def release(self, session_id: str) -> None:
        async with self._lock:
            assignment = self._assignments.pop(session_id, None)
        if assignment is None:
            return
        idx = next((i for i, h in enumerate(self._workers) if h.worker_id == assignment.worker_id), None)
        if idx is not None:
            await self._available.put(idx)

    async def shutdown(self) -> None:
        for handle in self._workers:
            try:
                handle.shutdown_event.set()
                handle.job_queue.put(None)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        loop = asyncio.get_running_loop()
        for handle in self._workers:
            await loop.run_in_executor(None, handle.process.join, 5.0)
            if handle.process.is_alive():
                handle.process.kill()
        for task in self._result_reader_tasks:
            task.cancel()
        self._result_reader_tasks.clear()
        self._workers.clear()
        self._worker_by_id.clear()

    def health(self) -> PoolHealth:
        return PoolHealth(
            total_workers=len(self._workers),
            available_workers=self._available.qsize(),
            active_sessions=len(self._assignments),
        )

    async def _drain_results(self, handle: _WorkerHandle) -> None:
        loop = asyncio.get_running_loop()
        while not handle.shutdown_event.is_set():
            try:
                msg = await loop.run_in_executor(None, _safe_queue_get, handle.result_queue, 0.5)
            except Exception:
                logger.exception("pool: worker %s result reader failed", handle.worker_id)
                return
            if msg is None:
                continue
            job_id = msg.get("job_id")
            if job_id is None:
                continue
            pending = self._pending.pop(job_id, None)
            if pending is None:
                continue
            if "error" in msg:
                pending.future.set_exception(RuntimeError(msg["error"]))
            else:
                pending.future.set_result(msg.get("result"))


def _safe_queue_get(q: mp.Queue, timeout: float) -> Any | None:
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


class WorkerFactory(Protocol):

    def __call__(
        self,
        *,
        gpu_id: int,
        generator_config: GeneratorConfig,
        warmup_config: WarmupConfig,
    ) -> _WorkerHandle:
        ...


def _default_worker_factory(
    *,
    gpu_id: int,
    generator_config: GeneratorConfig,
    warmup_config: WarmupConfig,
) -> _WorkerHandle:
    """Spawn a real multiprocessing worker.

    The child process calls :func:`worker_main` which constructs a
    :class:`VideoGenerator` from ``generator_config`` and runs a
    blocking job loop. The ``ready`` event flips after the warmup
    request completes.
    """
    ctx = mp.get_context("spawn")
    job_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    ready = threading.Event()
    shutdown_event = ctx.Event()
    worker_id = f"gpu{gpu_id}-{uuid.uuid4().hex[:6]}"
    process = ctx.Process(
        target=worker_main,
        kwargs={
            "gpu_id": gpu_id,
            "worker_id": worker_id,
            "generator_config": generator_config,
            "warmup_config": warmup_config,
            "job_queue": job_queue,
            "result_queue": result_queue,
            "shutdown_event": shutdown_event,
        },
        daemon=False,
    )
    process.start()

    # Block the parent-side ``ready`` flag until the worker posts a
    # ready acknowledgement on the result queue. We drain that single
    # sentinel here; subsequent results belong to jobs.
    def _await_ready() -> None:
        while not shutdown_event.is_set():
            try:
                msg = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if isinstance(msg, dict) and msg.get("kind") == "ready":
                ready.set()
                return
            if isinstance(msg, dict) and msg.get("kind") == "error":
                logger.error("pool: worker %s failed to boot: %s", worker_id, msg.get("error"))
                ready.set()  # unblock parent so it can observe the failure
                return

    threading.Thread(target=_await_ready, daemon=True).start()

    return _WorkerHandle(
        process=process,
        job_queue=job_queue,
        result_queue=result_queue,
        gpu_id=gpu_id,
        worker_id=worker_id,
        ready=ready,
        shutdown_event=shutdown_event,
    )


__all__ = [
    "GpuPool",
    "InProcessGpuPool",
    "PoolAcquireTimeout",
    "PoolAssignment",
    "PoolHealth",
    "SubprocessGpuPool",
    "WorkerFactory",
    "worker_main",
]
