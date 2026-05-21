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
from typing import Any

import torch

from fastvideo.eval.memory import clear_cache
from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import get_metric
from fastvideo.eval.types import MetricResult


class EvalWorker:
    """Owns metric replicas on one device. Single-GPU, single-sample.

    Per-sample metrics (``is_set_metric=False``) return one
    :class:`MetricResult` per ``evaluate`` call. Set metrics
    (``is_set_metric=True``) accumulate state on the worker's own
    instance and contribute nothing to the per-sample return — the
    Evaluator finalizes them after the pool drains.
    """

    def __init__(self, metric_names: list[str], device: str, *, compile: bool = False, pre_upload: bool = True) -> None:
        self._names = list(metric_names)
        self._device = device
        self._compile = compile
        self._pre_upload = pre_upload
        self._metrics: dict[str, BaseMetric] = {}
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
        # Skip pre_upload when no loaded metric actually runs on GPU
        # (e.g. an all-physics_iq run) — it would just pay a wasted
        # host→device→host round-trip per sample.
        self._any_needs_gpu = any(m.needs_gpu for m in self._metrics.values())
        self._unloaded = False

    def evaluate(self, **kwargs) -> dict[str, MetricResult]:
        """Score one already-decoded sample.

        Inputs (``video`` / ``reference`` paths, 5-D tensors) are
        decoded and normalized upstream by the :class:`VideoPool`. The
        worker only handles the optional pre-upload to its device and
        dispatches to each metric's ``compute`` or ``accumulate``.
        """
        if self._unloaded:
            raise RuntimeError("EvalWorker was unloaded; call reload() before evaluating.")

        sample = dict(kwargs)
        if self._pre_upload and self._any_needs_gpu:
            sample["video"] = _to_device(sample.get("video"), self._device)
            if "reference" in sample:
                sample["reference"] = _to_device(sample["reference"], self._device)

        results: dict[str, MetricResult] = {}
        for name, m in self._metrics.items():
            if m.is_set_metric:
                m.accumulate(sample)
            else:
                results[name] = m.compute(sample)
        return results

    def set_metrics(self) -> dict[str, BaseMetric]:
        """Return ``{name: instance}`` for set metrics on this worker."""
        return {n: m for n, m in self._metrics.items() if m.is_set_metric}

    def reset_set_metrics(self) -> None:
        """Clear accumulator state on every set metric."""
        for m in self._metrics.values():
            if m.is_set_metric:
                m.reset()

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
