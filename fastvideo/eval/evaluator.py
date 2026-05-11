"""User-facing scorer.

Layering (mirrors FastVideo's VideoGenerator → Worker pattern, but
in-process)::

    Evaluator               ← user-facing
      └── EvalWorker × N    ← single-GPU; owns metric replicas
      └── VideoPool         ← async path-→-tensor prefetch (per evaluate call)

The constructor builds one :class:`EvalWorker` per GPU and loads every
metric on every worker eagerly. :meth:`evaluate` is the single entry
point: pass kwargs for one sample, or pass a list of sample dicts to
fan-out across GPU replicas with pipelined decoding — same method,
return type follows the input shape.
"""
from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from fastvideo.eval.registry import (list_metrics, missing_dependencies, resolve_group)
from fastvideo.eval.types import MetricResult
from fastvideo.eval.worker import EvalWorker, add_pool_decode_ms
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class Evaluator:
    """Pre-initialized scorer for repeated evaluation.

    Parameters
    ----------
    metrics : list[str] | str
        Metric names, group prefixes (``"vbench"``), or ``"all"``.
    device : str
        Single-GPU device (e.g. ``"cuda:0"``). Ignored when *num_gpus* > 1.
    num_gpus : int
        Number of GPU replicas. Each gets its own :class:`EvalWorker`.
    compile : bool
        Apply :func:`torch.compile` to each metric's ``_model``.
    loader_threads : int
        Background decode threads in the :class:`VideoPool`. Default 1
        (hide decode behind compute). Bump for I/O-heavy benchmark sets
        where one loader can't keep up with the workers.
    prefetch_factor : int
        ``pool max_size = prefetch_factor * num_workers``. Default 2 —
        one sample being consumed, one prefetched per worker.
    pre_upload : bool
        If ``True`` (default), the worker uploads ``video`` /
        ``reference`` tensors to its device once per sample so every
        metric in the loop consumes the same GPU-resident tensor (no
        per-metric ``.to(self.device)`` traffic). Set ``False`` for
        training-time eval where the shared GPU-resident tensor would
        compete with the training step for VRAM — each metric then
        uploads its own copy as before.
    """

    def __init__(
        self,
        metrics: list[str] | str = "all",
        device: str = "cuda:0",
        num_gpus: int = 1,
        compile: bool = False,
        *,
        loader_threads: int = 1,
        prefetch_factor: int = 2,
        pre_upload: bool = True,
    ) -> None:
        names = _resolve_metric_names(metrics)
        if num_gpus > 1:
            self._workers = [
                EvalWorker(names, f"cuda:{i}", compile=compile, pre_upload=pre_upload) for i in range(num_gpus)
            ]
        else:
            self._workers = [EvalWorker(names, device, compile=compile, pre_upload=pre_upload)]
        self._loader_threads = max(1, loader_threads)
        self._prefetch_factor = max(1, prefetch_factor)

    @property
    def num_gpus(self) -> int:
        return len(self._workers)

    @property
    def metric_names(self) -> list[str]:
        return self._workers[0].metric_names

    def evaluate(
        self,
        samples: Iterable[dict] | None = None,
        **kwargs,
    ) -> dict[str, MetricResult] | list[dict[str, MetricResult]]:
        """Score one sample (kwargs form) or many samples (list form).

        ``video`` and ``reference`` may be either a pre-loaded
        ``(T, C, H, W)`` tensor or a path-like (``str`` / ``Path``).
        Paths in the list form are decoded asynchronously by a
        :class:`VideoPool` that runs alongside metric compute, hiding
        decode latency behind GPU work.

        One sample::

            ev.evaluate(video=tensor, text_prompt="...", fps=24.0)
            ev.evaluate(video="path/to/clip.mp4", fps=24.0)

        Many samples — pipelined decode + work-stealing across replicas::

            ev.evaluate(samples=[
                {"video": "a.mp4", "reference": "ref_a.mp4"},
                {"video": "b.mp4", "reference": "ref_b.mp4"},
                ...
            ])

        Multi-GPU dispatch fires automatically when ``num_gpus > 1`` and
        the list form is used: every worker runs a consumer thread,
        pulling decoded samples from the shared pool as it frees up.
        The kwargs form always runs on worker 0 with no pool overhead.
        """
        if samples is None:
            return self._workers[0].evaluate(**kwargs)

        samples = list(samples)
        if not samples:
            return []
        return self._evaluate_with_pool(samples)

    def _evaluate_with_pool(self, samples: list[dict]) -> list[dict[str, MetricResult]]:
        """Pipelined dispatch: ``VideoPool`` prefetches decoded samples;
        consumers (one per worker) pop them and run metrics.

        Decode order in the pool is non-deterministic — each pool item
        carries its original input index so results are written back in
        input order.
        """
        from fastvideo.eval.pool import VideoPool

        n_workers = len(self._workers)
        max_size = self._prefetch_factor * n_workers
        results: list[Any] = [None] * len(samples)

        with VideoPool(samples, loader_threads=self._loader_threads, max_size=max_size) as pool:
            if n_workers == 1:
                # Single-GPU: this thread is the consumer.
                while True:
                    item = pool.get()
                    if item is None:
                        break
                    idx, decoded = item
                    results[idx] = self._workers[0].evaluate(**decoded)
            else:
                # Multi-GPU: each worker runs its own consumer thread,
                # pulling from the shared pool (work-stealing).
                threads: list[threading.Thread] = []
                for w in self._workers:
                    t = threading.Thread(
                        target=self._consumer_loop,
                        args=(w, pool, results),
                        daemon=True,
                    )
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()

            # Attribute pool-side decode ms to the same global counter
            # the worker uses so ``pop_timings()`` returns total decode
            # time regardless of where the decode happened.
            add_pool_decode_ms(pool.decode_ms_total)

        return results

    @staticmethod
    def _consumer_loop(worker: EvalWorker, pool: Any, results: list) -> None:
        while True:
            item = pool.get()
            if item is None:
                return
            idx, decoded = item
            results[idx] = worker.evaluate(**decoded)

    def release_cuda_memory(self) -> None:
        """Free CUDA caches on every replica without dropping models."""
        for w in self._workers:
            w.release_cuda_memory()

    def unload(self) -> None:
        """Drop metric refs on every replica. Reverse with :meth:`reload`."""
        for w in self._workers:
            w.unload()

    def reload(self) -> None:
        """Rebuild metrics dropped by :meth:`unload`."""
        for w in self._workers:
            w.reload()

    def shutdown(self) -> None:
        """No-op kept for API compatibility.

        Earlier versions of this class held a long-lived
        ``ThreadPoolExecutor`` for multi-GPU round-robin dispatch and
        needed an explicit shutdown to drain it. The current design
        builds and tears down a :class:`VideoPool` per ``evaluate``
        call, so there's no long-lived state to release here.
        """


def create_evaluator(
    metrics: list[str] | str = "all",
    device: str = "cuda:0",
    num_gpus: int = 1,
    compile: bool = False,
) -> Evaluator:
    return Evaluator(metrics=metrics, device=device, num_gpus=num_gpus, compile=compile)


def _resolve_metric_names(metrics: list[str] | str) -> list[str]:
    """Resolve metric names, supporting groups (``"vbench"``) and ``"all"``.

    Group / ``"all"`` selectors silently skip metrics whose declared
    dependencies aren't importable in this environment, with a single
    warning per skipped metric. Explicit names (e.g. ``"vbench.color"``)
    always pass through unchanged — the missing dep then surfaces as
    :class:`ImportError` at construction time, which is what the user
    asked for.
    """
    if metrics == "all":
        return _filter_satisfied(list_metrics(), context="all")
    if isinstance(metrics, str):
        metrics = [metrics]

    seen: set[str] = set()
    names: list[str] = []
    for m in metrics:
        group = resolve_group(m)
        candidates = _filter_satisfied(group, context=m) if group is not None else [m]
        for n in candidates:
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names


def _filter_satisfied(names: list[str], *, context: str) -> list[str]:
    """Drop metrics with missing deps from a group expansion."""
    keep: list[str] = []
    for n in names:
        missing = missing_dependencies(n)
        if missing:
            logger.warning(
                "eval: skipping %s in group '%s'; missing dependency: %s. "
                "Install instructions: pass the metric name explicitly to see them.", n, context, ", ".join(missing))
            continue
        keep.append(n)
    return keep
