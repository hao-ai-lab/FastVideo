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

import torch

from fastvideo.eval.types import Video

_SENTINEL = object()


class _DecodeError:
    """Marker pushed onto the ready queue when a loader thread raises.

    The consumer re-raises in its own thread, surfacing the error to
    the caller of ``Evaluator.evaluate`` instead of hanging on
    ``_ready_q.get()`` forever.
    """

    __slots__ = ("exc", )

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


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

    def get(self) -> tuple[int, dict] | None:
        """Pop the next decoded ``(idx, sample)``.

        Returns ``None`` when all input samples have been consumed.
        Polls in 0.1 s slices so extra consumer threads (when
        ``len(samples) < num_workers``) wake up periodically to
        re-check ``_consumed`` and exit cleanly — without the poll,
        a blocking ``_ready_q.get(timeout=None)`` deadlocks because
        the loaders are already done. Re-raises any exception caught
        in a loader thread on the consumer's stack so callers don't
        hang on a dead loader. Thread-safe: multiple consumer threads
        may share one pool.
        """
        while True:
            with self._consume_lock:
                if self._consumed >= len(self._samples):
                    return None
            try:
                item = self._ready_q.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._consume_lock:
                self._consumed += 1
            idx, payload = item
            if isinstance(payload, _DecodeError):
                raise payload.exc
            return item

    def _loader_loop(self) -> None:
        while not self._stop.is_set():
            item = self._task_q.get()
            if item is _SENTINEL:
                return
            idx, sample = item
            try:
                decoded: Any = self._decode(sample)
            except BaseException as exc:  # noqa: BLE001 — forward to consumer
                self._ready_q.put((idx, _DecodeError(exc)))
                continue
            # Blocking put: under normal flow the consumer drains the
            # queue; under shutdown ``__exit__`` drains it for us. A
            # timeout would silently drop samples and hang the consumer.
            self._ready_q.put((idx, decoded))

    def _decode(self, sample: dict) -> dict:
        """Materialize and normalize ``video`` / ``reference`` entries.

        * ``Video`` instance → populate ``.frames`` via decode.
        * ``str`` / ``Path`` under ``video`` / ``reference`` → decoded
          ``(T, C, H, W)`` tensor.
        * ``(1, T, C, H, W)`` tensor under ``video`` / ``reference`` →
          squeezed to ``(T, C, H, W)`` (back-compat with callers that
          still pass a leading batch dim).
        * Everything else passes through unchanged.
        """
        from fastvideo.eval.io.video import load_video

        out = dict(sample)
        for key, val in sample.items():
            if isinstance(val, Video):
                if val.frames is None and val.source is not None:
                    val.frames = load_video(val.source)
                out[key] = val
            elif key in ("video", "reference"):
                if isinstance(val, str | Path):
                    out[key] = load_video(str(val))
                elif isinstance(val, torch.Tensor) and val.dim() == 5 and val.shape[0] == 1:
                    out[key] = val.squeeze(0)
        return out
