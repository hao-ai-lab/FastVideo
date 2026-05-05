from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from fastvideo.eval.types import MetricResult


class BaseMetric(ABC):
    """Abstract base class for all eval metrics.

    Subclasses must implement :meth:`compute`. Optionally override
    :meth:`setup` to eagerly load models.

    Metrics that need to chunk along the time dimension (frames or frame
    pairs) for memory reasons should hardcode their own chunk size in
    ``__init__`` (see ``optical_flow`` for the canonical example). The
    :class:`EvalWorker` above never chunks across the batch dim because
    eval is always B=1: one video per :meth:`Evaluator.evaluate` call.
    """

    name: str = ""
    requires_reference: bool = True
    higher_is_better: bool = True
    dependencies: list[str] = []
    needs_gpu: bool = False
    backbone: str | None = None

    # Default time-dim chunk size for metrics that batch internally over
    # frames or frame-pairs. Override in subclass __init__ if needed
    # (see ``optical_flow``, ``motion_smoothness``, ``dynamic_degree``).
    _chunk_size: int | None = None

    def __init__(self) -> None:
        self._device: torch.device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: str | torch.device) -> BaseMetric:
        """Move metric (and its internal models) to *device*."""
        self._device = torch.device(device)
        return self

    def setup(self) -> None:
        """Eagerly load models. Called once by :class:`EvalWorker`."""
        pass

    def _skip(self, sample: dict, reason: str) -> list[MetricResult]:
        """Return one skipped result per sample row (score=None)."""
        B = sample["video"].shape[0] if "video" in sample else 1
        return [MetricResult(name=self.name, score=None,
                             details={"skipped": reason}) for _ in range(B)]

    @abstractmethod
    def compute(self, sample: dict) -> list[MetricResult]:
        """Compute the metric on a single-row sample.

        ``sample["video"]`` is ``(1, T, C, H, W)``. Returns a one-element
        list of :class:`MetricResult` (the leading 1 is preserved for
        backwards compatibility with metrics that loop ``for b in range(B)``).

        If required inputs are missing, return skipped results via
        ``self._skip(sample, reason)`` instead of raising.
        """
        ...
