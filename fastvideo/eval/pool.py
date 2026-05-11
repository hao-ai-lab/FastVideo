"""Centralized prefetcher for the Evaluator.

Hides video-decode latency (CPU + disk I/O) behind metric compute (GPU)
by running a small thread pool of decoders that fill a bounded queue.

A single :class:`VideoPool` is owned by the Evaluator for the duration
of one ``evaluate(samples=...)`` call. Workers consume from it via
:meth:`VideoPool.get`; decode order is non-deterministic (whichever
loader finishes first wins), but each yielded item carries its
original input index so the consumer can write into a result list at
the right slot.

Pool sizing: ``max_size = prefetch_factor * num_workers``. With the
default ``prefetch_factor=2``, that's two decoded samples in flight
per worker (one being consumed, one ready).
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

from fastvideo.eval.types import Video

_SENTINEL = object()


class VideoPool:
    """Bounded prefetch queue feeding decoded samples to consumers.

    Use as a context manager so loader threads are always cleaned up::

        with VideoPool(samples, loader_threads=1, max_size=4) as pool:
            while True:
                item = pool.get()
                if item is None:
                    break
                idx, decoded = item
                results[idx] = worker.evaluate(**decoded)
    """

    def __init__(
        self,
        samples: list[dict],
        *,
        loader_threads: int = 1,
        max_size: int = 4,
    ) -> None:
        if loader_threads < 1:
            raise ValueError("loader_threads must be >= 1")
        self._samples = samples
        self._loader_threads_n = loader_threads
        self._max_size = max(max_size, 1)

        self._task_q: queue.Queue = queue.Queue()
        self._ready_q: queue.Queue = queue.Queue(maxsize=self._max_size)
        self._loaders: list[threading.Thread] = []
        self._stop = threading.Event()

        self._consumed = 0
        self._consume_lock = threading.Lock()

        self._decode_ms_total = 0.0
        self._decode_lock = threading.Lock()

    # --- context-manager lifecycle ---

    def __enter__(self) -> VideoPool:
        for idx, sample in enumerate(self._samples):
            self._task_q.put((idx, sample))
        # One sentinel per loader so each thread can exit cleanly.
        for _ in range(self._loader_threads_n):
            self._task_q.put(_SENTINEL)
        for _ in range(self._loader_threads_n):
            t = threading.Thread(target=self._loader_loop, daemon=True)
            t.start()
            self._loaders.append(t)
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        # Drain the ready queue so any blocked-on-put loader unblocks.
        while True:
            try:
                self._ready_q.get_nowait()
            except queue.Empty:
                break
        for t in self._loaders:
            t.join(timeout=5.0)

    # --- consumer API ---

    def get(self, timeout: float | None = None) -> tuple[int, dict] | None:
        """Pop the next decoded ``(idx, sample)``.

        Returns ``None`` when all input samples have been consumed.
        Thread-safe: multiple consumer threads may share one pool.
        """
        with self._consume_lock:
            if self._consumed >= len(self._samples):
                return None
        try:
            item = self._ready_q.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._consume_lock:
            self._consumed += 1
        return item

    @property
    def decode_ms_total(self) -> float:
        with self._decode_lock:
            return self._decode_ms_total

    # --- loader internals ---

    def _loader_loop(self) -> None:
        while not self._stop.is_set():
            item = self._task_q.get()
            if item is _SENTINEL:
                return
            idx, sample = item
            decoded = self._decode(sample)
            try:
                self._ready_q.put((idx, decoded), timeout=10.0)
            except queue.Full:
                # Stop set during shutdown; drop and exit.
                return

    def _decode(self, sample: dict) -> dict:
        """Walk a sample dict, materialize any path-shaped video values.

        Two recognised shapes:
        - A :class:`Video` instance — populate ``.frames`` (lazy decode).
        - A bare path under ``video`` / ``reference`` — load to
          ``(T, C, H, W)`` tensor (back-compat with existing callers).

        Anything else (audio paths, scalars, dicts, tensors) passes
        through unchanged.
        """
        from fastvideo.eval.io.video import load_video

        t0 = time.perf_counter()
        out = dict(sample)
        for key, val in sample.items():
            if isinstance(val, Video):
                if val.frames is None and val.source is not None:
                    val.frames = load_video(val.source)
                out[key] = val
            elif key in ("video", "reference") and isinstance(val, str | Path):
                out[key] = load_video(str(val))
        with self._decode_lock:
            self._decode_ms_total += (time.perf_counter() - t0) * 1000.0
        return out
