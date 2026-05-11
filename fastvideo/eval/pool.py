"""Async path-→-tensor prefetcher for the Evaluator.

Hides video-decode latency behind metric compute by running a small
thread pool of decoders that fill a bounded queue. One :class:`VideoPool`
is owned by the Evaluator per ``evaluate(samples=...)`` call; workers
consume via :meth:`VideoPool.get`. Decode order is non-deterministic;
each yielded item carries its original input index so consumers can
write back into a result list in input order.

Pool sizing: ``max_size = prefetch_factor * num_workers``.
"""
from __future__ import annotations

import queue
import threading
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

    def __enter__(self) -> VideoPool:
        for idx, sample in enumerate(self._samples):
            self._task_q.put((idx, sample))
        for _ in range(self._loader_threads_n):
            self._task_q.put(_SENTINEL)
        for _ in range(self._loader_threads_n):
            t = threading.Thread(target=self._loader_loop, daemon=True)
            t.start()
            self._loaders.append(t)
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        # Drain ready queue so any blocked-on-put loader unblocks.
        while True:
            try:
                self._ready_q.get_nowait()
            except queue.Empty:
                break
        for t in self._loaders:
            t.join(timeout=5.0)

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
                return

    def _decode(self, sample: dict) -> dict:
        """Materialize any path-shaped video values in *sample*.

        Recognises ``Video`` instances (populates ``.frames``) and bare
        path strings under ``video`` / ``reference``. Other entries pass
        through unchanged.
        """
        from fastvideo.eval.io.video import load_video

        out = dict(sample)
        for key, val in sample.items():
            if isinstance(val, Video):
                if val.frames is None and val.source is not None:
                    val.frames = load_video(val.source)
                out[key] = val
            elif key in ("video", "reference") and isinstance(val, str | Path):
                out[key] = load_video(str(val))
        return out
