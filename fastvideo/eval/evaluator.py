"""User-facing scorer.

Layering (mirrors FastVideo's VideoGenerator → Worker pattern, but
in-process)::

    Evaluator               ← user-facing; round-robins samples across workers
      └── EvalWorker × N    ← single-GPU; owns metric replicas

The constructor builds one :class:`EvalWorker` per GPU and loads every
metric on every worker eagerly. :meth:`evaluate` is the single entry
point: pass kwargs for one sample, or pass a list of sample dicts to
fan-out across GPU replicas — same method, return type follows the
input shape.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from fastvideo.eval.registry import list_metrics, resolve_group
from fastvideo.eval.types import MetricResult
from fastvideo.eval.worker import EvalWorker


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
    """

    def __init__(
        self,
        metrics: list[str] | str = "all",
        device: str = "cuda:0",
        num_gpus: int = 1,
        compile: bool = False,
    ) -> None:
        names = _resolve_metric_names(metrics)
        if num_gpus > 1:
            self._workers = [
                EvalWorker(names, f"cuda:{i}", compile=compile)
                for i in range(num_gpus)
            ]
        else:
            self._workers = [EvalWorker(names, device, compile=compile)]
        self._pool = (ThreadPoolExecutor(max_workers=num_gpus)
                      if num_gpus > 1 else None)

    @property
    def num_gpus(self) -> int:
        return len(self._workers)

    @property
    def metric_names(self) -> list[str]:
        return list(self._workers[0]._metrics.keys())

    def evaluate(
        self,
        samples: Iterable[dict] | None = None,
        **kwargs,
    ) -> dict[str, MetricResult] | list[dict[str, MetricResult]]:
        """Score one sample (kwargs form) or many samples (list form).

        One sample::

            ev.evaluate(video=tensor, text_prompt=[...], fps=24.0)

        Many samples — fan out across GPU replicas, results in input order::

            ev.evaluate([
                {"video": v1, "text_prompt": [...], "fps": 24.0},
                {"video": v2, "text_prompt": [...], "fps": 24.0},
                ...
            ])

        Multi-GPU dispatch fires automatically iff ``num_gpus > 1`` *and*
        the list form is used. The kwargs form always runs on worker 0;
        if you have a single sample but multiple GPUs, wrap it in a
        one-element list to use the pool, or just accept that a single
        call uses one GPU — that's fine.
        """
        if samples is None:
            return self._workers[0].evaluate(**kwargs)

        samples = list(samples)
        if self._pool is None or len(samples) <= 1:
            return [self._workers[0].evaluate(**s) for s in samples]

        n = len(self._workers)
        # Round-robin: worker i handles samples i, i+n, i+2n, ...
        futures = [
            self._pool.submit(self._workers[idx % n].evaluate, **sample)
            for idx, sample in enumerate(samples)
        ]
        return [f.result() for f in futures]

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
        """Tear down the worker thread pool. Idempotent."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def create_evaluator(
    metrics: list[str] | str = "all",
    device: str = "cuda:0",
    num_gpus: int = 1,
    compile: bool = False,
) -> Evaluator:
    return Evaluator(metrics=metrics, device=device, num_gpus=num_gpus,
                     compile=compile)


def _resolve_metric_names(metrics: list[str] | str) -> list[str]:
    """Resolve metric names, supporting groups (``"vbench"``) and ``"all"``."""
    if metrics == "all":
        return list_metrics()
    if isinstance(metrics, str):
        metrics = [metrics]

    seen: set[str] = set()
    names: list[str] = []
    for m in metrics:
        group = resolve_group(m)
        for n in (group if group is not None else [m]):
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names
