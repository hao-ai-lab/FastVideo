from __future__ import annotations

import torch

from fastvideo.eval.types import MetricResult


class BaseMetric:
    """Abstract base class for all eval metrics.

    Two execution shapes:

    * **Per-sample** (``is_set_metric=False``, default) — implement
      :meth:`compute`. The Evaluator calls it once per input sample and
      returns one :class:`MetricResult` per sample.

    * **Set-vs-set** (``is_set_metric=True``) — implement
      :meth:`accumulate` (called once per sample to buffer features)
      and :meth:`finalize` (called once after all samples to compute
      the corpus-level result). Use :meth:`reset` to clear buffers and
      :meth:`merge_from` to fold multi-GPU per-worker state together.

    Optionally override :meth:`setup` to eagerly load models. Metrics
    that chunk along the time dim for memory hardcode their own chunk
    size in ``__init__`` (see ``optical_flow`` for the canonical
    example). Eval always processes one video per
    :meth:`Evaluator.evaluate` call; ``compute`` / ``accumulate``
    receive a single sample, not a batch.
    """

    name: str = ""
    requires_reference: bool = True
    higher_is_better: bool = True
    dependencies: list[str] = []
    needs_gpu: bool = False
    backbone: str | None = None
    is_set_metric: bool = False

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

    def setup(self) -> None:  # noqa: B027 - intentionally optional override
        """Eagerly load models. Called once by :class:`EvalWorker`.

        Default is a no-op; metrics with no eager state (pixel math,
        closed-form ops) inherit this. Override only if your metric
        needs to load weights.
        """

    def _skip(self, sample: dict, reason: str) -> MetricResult:
        """Return a skipped result (``score=None`` + reason in details)."""
        return MetricResult(name=self.name, score=None, details={"skipped": reason})

    def compute(self, sample: dict) -> MetricResult:
        """Per-sample metrics: compute the score for one sample.

        ``sample["video"]`` is ``(T, C, H, W)`` float in ``[0, 1]``.
        ``sample["reference"]`` (if used) has the same shape. Return
        ``self._skip(sample, reason)`` for missing inputs.
        """
        raise NotImplementedError(f"{type(self).__name__}.compute is not implemented")

    # --- set-vs-set protocol (only invoked when is_set_metric=True) ---

    def reset(self) -> None:  # noqa: B027 - intentionally optional override
        """Clear accumulator state at the start of each evaluate() call."""

    def accumulate(self, sample: dict) -> None:
        """Buffer per-sample features for a corpus-level metric."""
        raise NotImplementedError(f"{type(self).__name__}.accumulate is not implemented")

    def finalize(self) -> MetricResult:
        """Compute the corpus-level result from buffered state."""
        raise NotImplementedError(f"{type(self).__name__}.finalize is not implemented")

    def merge_from(self, other: BaseMetric) -> None:  # noqa: B027 - intentionally optional override
        """Multi-GPU: fold another worker's accumulator state into this one."""
