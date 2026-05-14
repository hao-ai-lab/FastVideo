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
from fastvideo.eval.types import EvalResults, MetricResult
from fastvideo.eval.worker import EvalWorker
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
        When ``True`` (default), the worker performs a single
        host→device upload of ``video`` / ``reference`` per sample
        before the metric loop, and every metric reads from that
        shared GPU-resident tensor. Without it, each metric pays its
        own ``.to(self.device)`` — N transfers of the same clip for N
        metrics, which dominates at high resolution. Set ``False`` for
        training-time eval, where keeping a clip resident on GPU
        across the metric loop would fight the training step for VRAM.
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
    ) -> dict[str, MetricResult] | EvalResults:
        """Score one sample (kwargs form) or many samples (list form).

        Both forms go through the same :class:`VideoPool` pipeline;
        ``video`` / ``reference`` paths are decoded asynchronously,
        ``(1, T, C, H, W)`` tensors are squeezed.

        One sample::

            ev.evaluate(video=tensor, text_prompt="...", fps=24.0)
            ev.evaluate(video="path/to/clip.mp4", fps=24.0)

        Returns a ``dict[str, MetricResult]``.

        Many samples — pipelined decode + work-stealing across replicas::

            ev.evaluate(samples=[
                {"video": "a.mp4", "reference": "ref_a.mp4"},
                {"video": "b.mp4", "reference": "ref_b.mp4"},
                ...
            ])

        Returns an :class:`EvalResults` (list-of-dict subclass): per-sample
        dicts in input order, with set-metric scores under ``.corpus``.
        """
        single = samples is None
        sample_list: list[dict] = [kwargs] if samples is None else list(samples)
        if not sample_list:
            return EvalResults(samples=[], corpus={})

        if single and any(m.is_set_metric for m in self._workers[0].set_metrics().values()):
            # Set metrics need a population. A single sample can't produce a
            # meaningful corpus result, and silently discarding it (return
            # ``per_sample[0]`` only) hides the no-op. Force the list form.
            raise ValueError("Set-vs-set metrics require samples=[...] with >=2 entries; "
                             "the kwargs form (single sample) cannot produce a corpus "
                             "result. Registered set metrics: "
                             f"{sorted(self._workers[0].set_metrics())}")

        per_sample, corpus = self._run(sample_list)

        if single:
            return per_sample[0]
        return EvalResults(samples=per_sample, corpus=corpus)

    def _run(self, samples: list[dict]) -> tuple[list[dict[str, MetricResult]], dict[str, MetricResult]]:
        """Pool-driven sample pipeline + set-metric finalize.

        Returns ``(per_sample_results, corpus_results)``.
        """
        from fastvideo.eval.pool import VideoPool

        # Reset every worker's set-metric buffers — per-call isolation.
        for w in self._workers:
            w.reset_set_metrics()

        n_workers = len(self._workers)
        max_size = self._prefetch_factor * n_workers
        per_sample: list[Any] = [None] * len(samples)

        with VideoPool(samples, loader_threads=self._loader_threads, max_size=max_size) as pool:
            if n_workers == 1:
                while True:
                    item = pool.get()
                    if item is None:
                        break
                    idx, decoded = item
                    per_sample[idx] = self._workers[0].evaluate(**decoded)
            else:
                # Multi-GPU: every worker drains the shared pool (work-stealing).
                errors: list[BaseException] = []
                threads: list[threading.Thread] = []
                for w in self._workers:
                    t = threading.Thread(target=self._consumer_loop, args=(w, pool, per_sample, errors), daemon=True)
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()
                if errors:
                    raise errors[0]

        # Finalize set metrics. With multiple workers, fold per-worker
        # accumulator state into worker 0 first, then finalize once.
        corpus: dict[str, MetricResult] = {}
        base_set = self._workers[0].set_metrics()
        if base_set:
            for w in self._workers[1:]:
                for name, m in w.set_metrics().items():
                    base_set[name].merge_from(m)
            corpus = {name: m.finalize() for name, m in base_set.items()}

        return per_sample, corpus

    @staticmethod
    def _consumer_loop(worker: EvalWorker, pool: Any, results: list, errors: list) -> None:
        try:
            while True:
                item = pool.get()
                if item is None:
                    return
                idx, decoded = item
                results[idx] = worker.evaluate(**decoded)
        except BaseException as e:  # noqa: BLE001 — surface to parent thread via shared list
            errors.append(e)

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
        """No-op; kept for API compatibility with older callers."""


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
