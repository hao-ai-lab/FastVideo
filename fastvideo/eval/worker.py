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

import torch

from fastvideo.eval.memory import clear_cache
from fastvideo.eval.registry import get_metric
from fastvideo.eval.types import MetricResult


class EvalWorker:
    """Owns metric replicas on one device. Single-GPU, single-sample."""

    def __init__(self, metric_names: list[str], device: str, *, compile: bool = False) -> None:
        self._names = list(metric_names)
        self._device = device
        self._compile = compile
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
        """
        if self._unloaded:
            raise RuntimeError("EvalWorker was unloaded; call reload() before evaluating.")

        sample = dict(kwargs)
        sample["video"] = _resolve_video_input(sample.get("video"))
        if "reference" in sample:
            sample["reference"] = _resolve_video_input(sample["reference"])

        results: dict[str, MetricResult] = {}
        for name, m in self._metrics.items():
            results[name] = m.compute(sample)
        return results

    def release_cuda_memory(self) -> None:
        """Free CUDA caches without dropping models."""
        clear_cache()
        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def unload(self) -> None:
        """Drop metric refs so models become GC-able. Reverse with reload()."""
        self._metrics = {}
        self._unloaded = True
        self.release_cuda_memory()

    def reload(self) -> None:
        """Rebuild metrics dropped by :meth:`unload`."""
        if self._unloaded:
            self._load()


def _resolve_video_input(value):
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
    if isinstance(value, (str, Path)):
        from fastvideo.eval.io.video import load_video
        return load_video(str(value))
    if isinstance(value, torch.Tensor) and value.dim() == 5 and value.shape[0] == 1:
        return value.squeeze(0)
    return value
