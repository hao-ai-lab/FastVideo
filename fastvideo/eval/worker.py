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

import contextlib
from pathlib import Path
from typing import Any

import torch

from fastvideo.eval.memory import clear_cache
from fastvideo.eval.registry import get_metric
from fastvideo.eval.types import MetricResult


class EvalWorker:
    """Owns metric replicas on one device. Single-GPU, single-sample.

    Metrics receive one sample per ``compute(sample)`` call: scalar
    values, not list-wrapped or batch-dim-prefixed. See
    :class:`Evaluator` for ``pre_upload`` semantics.
    """

    def __init__(self, metric_names: list[str], device: str, *, compile: bool = False, pre_upload: bool = True) -> None:
        self._names = list(metric_names)
        self._device = device
        self._compile = compile
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

        ``video`` / ``reference`` may be a ``(T, C, H, W)`` tensor or a
        path-like (``str`` / ``Path``). Paths are decoded here so the
        dispatcher can queue cheap strings. A ``(1, T, C, H, W)`` tensor
        is unwrapped to ``(T, C, H, W)`` for back-compat.
        """
        if self._unloaded:
            raise RuntimeError("EvalWorker was unloaded; call reload() before evaluating.")

        sample = dict(kwargs)
        sample["video"] = _resolve_video_input(sample.get("video"))
        if "reference" in sample:
            sample["reference"] = _resolve_video_input(sample["reference"])
        if self._pre_upload:
            sample["video"] = _to_device(sample.get("video"), self._device)
            if "reference" in sample:
                sample["reference"] = _to_device(sample["reference"], self._device)

        results: dict[str, MetricResult] = {}
        for name, m in self._metrics.items():
            results[name] = m.compute(sample)
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


def _to_device(value: Any, device: str | torch.device) -> Any:
    """Move *value* to *device* if it's a tensor not already there."""
    if value is None or not isinstance(value, torch.Tensor):
        return value
    target = torch.device(device)
    if value.device == target:
        return value
    return value.to(target, non_blocking=True)


def _resolve_video_input(value: Any) -> Any:
    """Normalize a sample's ``video`` / ``reference`` field for metrics.

    Paths → decoded ``(T, C, H, W)`` tensor; ``(1, T, C, H, W)`` →
    squeezed to ``(T, C, H, W)``; everything else returned untouched.
    """
    if value is None:
        return None
    if isinstance(value, str | Path):
        from fastvideo.eval.io.video import load_video
        return load_video(str(value))
    if isinstance(value, torch.Tensor) and value.dim() == 5 and value.shape[0] == 1:
        return value.squeeze(0)
    return value
