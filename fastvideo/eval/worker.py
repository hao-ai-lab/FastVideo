"""EvalWorker: a single-GPU bag of metric replicas.

One ``EvalWorker`` per GPU. Holds an instance of every requested metric on
its own device, scores one sample at a time. Stateless across calls — the
worker never sees more than one ``evaluate(...)`` invocation at a time
(the parent :class:`Evaluator` ensures this by handing the worker to one
thread at a time).

This mirrors FastVideo's ``Worker`` layer under :class:`VideoGenerator`,
but in-process (threads, not processes) — eval metrics are independent
and need no NCCL / TP / SP / distributed init, so process isolation is
unnecessary overhead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fastvideo.eval.memory import clear_cache
from fastvideo.eval.registry import get_metric
from fastvideo.eval.types import MetricResult
import contextlib


class EvalWorker:
    """Owns metric replicas on one device. Single-GPU, single-sample."""

    def __init__(self, metric_names: list[str], device: str, *, compile: bool = False, pre_upload: bool = True) -> None:
        self._names = list(metric_names)
        self._device = device
        self._compile = compile
        # Pre-upload ``video``/``reference`` to the worker's device once
        # so every metric for the same sample shares a single GPU-resident
        # tensor. Set ``pre_upload=False`` for training-time eval contexts
        # where holding the input tensor on GPU across the metric loop
        # competes with the training-step working set; in that case each
        # metric uploads its own copy as before.
        self._pre_upload = pre_upload
        self._metrics: dict = {}
        self._unloaded = False
        self._load()

    @property
    def device(self) -> str:
        return self._device

    @property
    def metric_names(self) -> list[str]:
        """Names of the metrics this worker owns, in load order."""
        return list(self._metrics.keys())

    def _load(self) -> None:
        for name in self._names:
            m = get_metric(name)
            m.to(self._device)
            m.setup()
            if self._compile and getattr(m, "_model", None) is not None:
                m._model = torch.compile(m._model)
            self._metrics[name] = m
        self._unloaded = False

    def evaluate(self, **kwargs) -> dict[str, MetricResult]:
        """Score one sample.

        ``video`` may be a ``(T, C, H, W)`` tensor or a path-like
        (``str`` / ``Path``) — paths are loaded inside this method so
        the dispatcher can hold a queue of cheap path strings instead
        of fully-decoded tensors. ``reference`` follows the same rule.

        A ``(1, T, C, H, W)`` tensor is also accepted for back-compat
        and gets unwrapped to ``(T, C, H, W)`` before reaching metrics.

        Stage timings (``decode_ms``, ``compute_ms``) are accumulated on
        thread-local counters and zeroed on each call. Read via
        :func:`pop_timings` immediately after the call.
        """
        if self._unloaded:
            raise RuntimeError("EvalWorker was unloaded; call reload() before evaluating.")

        import time
        sample = dict(kwargs)
        t0 = time.perf_counter()
        sample["video"] = _resolve_video_input(sample.get("video"))
        if "reference" in sample:
            sample["reference"] = _resolve_video_input(sample["reference"])
        if self._pre_upload:
            sample["video"] = _to_device(sample.get("video"), self._device)
            if "reference" in sample:
                sample["reference"] = _to_device(sample["reference"], self._device)
        t1 = time.perf_counter()

        results: dict[str, MetricResult] = {}
        for name, m in self._metrics.items():
            results[name] = m.compute(sample)
        t2 = time.perf_counter()

        _record_timing(decode_ms=(t1 - t0) * 1000.0, compute_ms=(t2 - t1) * 1000.0)
        return results

    def release_cuda_memory(self) -> None:
        """Free CUDA caches without dropping models."""
        clear_cache()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.ipc_collect()

    def unload(self) -> None:
        """Drop metric refs so models become GC-able. Reverse with reload()."""
        self._metrics = {}
        self._unloaded = True
        self.release_cuda_memory()

    def reload(self) -> None:
        """Rebuild metrics dropped by :meth:`unload`."""
        if self._unloaded:
            self._load()


_TIMINGS: dict[str, float] = {"decode_ms": 0.0, "compute_ms": 0.0, "n": 0}


def _record_timing(*, decode_ms: float, compute_ms: float) -> None:
    """Accumulate per-call decode and compute time into a process-global
    counter. Read + zeroed via :func:`pop_timings`. Used by
    ``scripts/eval/bench_pipeline.py``.
    """
    _TIMINGS["decode_ms"] += decode_ms
    _TIMINGS["compute_ms"] += compute_ms
    _TIMINGS["n"] += 1


def add_pool_decode_ms(ms: float) -> None:
    """Attribute pool-side decode time to the same global counter.

    Pool threads call this when they finish decoding a sample. Keeps
    ``pop_timings()`` returning total decode time regardless of where
    the decode physically happened.
    """
    _TIMINGS["decode_ms"] += ms


def pop_timings() -> dict[str, float]:
    """Snapshot and zero the timing counters."""
    snapshot = dict(_TIMINGS)
    _TIMINGS["decode_ms"] = 0.0
    _TIMINGS["compute_ms"] = 0.0
    _TIMINGS["n"] = 0
    return snapshot


def _to_device(value: Any, device: str | torch.device) -> Any:
    """Move a video tensor to *device* if it isn't already there.

    No-op for ``None`` or non-tensor values. Uses ``non_blocking=True``
    so the host-side memcpy and device DMA can overlap with subsequent
    CPU work (pinning is opportunistic; PyTorch will pin internally for
    pageable tensors).
    """
    if value is None or not isinstance(value, torch.Tensor):
        return value
    target = torch.device(device)
    if value.device == target:
        return value
    return value.to(target, non_blocking=True)


def _resolve_video_input(value: Any) -> Any:
    """Normalize a sample's ``video`` / ``reference`` field for metrics.

    * ``str`` / ``Path`` → decoded ``(T, C, H, W)`` tensor via
      :func:`fastvideo.eval.io.video.load_video`. Decoding happens in
      the worker thread so the dispatcher can keep paths queued
      instead of full tensors.
    * ``(1, T, C, H, W)`` tensor → squeezed to ``(T, C, H, W)``
      (back-compat with callers that still pass the leading batch dim).
    * anything else → returned untouched.
    """
    if value is None:
        return None
    if isinstance(value, str | Path):
        from fastvideo.eval.io.video import load_video
        return load_video(str(value))
    if isinstance(value, torch.Tensor) and value.dim() == 5 and value.shape[0] == 1:
        return value.squeeze(0)
    return value
