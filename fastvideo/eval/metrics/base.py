from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from fastvideo.eval.types import MetricResult


class BaseMetric(ABC):
    """Abstract base class for all wm-eval metrics.

    Subclasses must implement :meth:`compute`.  Optionally override
    :meth:`setup` to eagerly load models, and :meth:`trial_forward`
    to support auto-calibration of batch sizes.

    ``batch_unit`` declares what the metric batches over internally:
    ``"video"`` (whole videos, B dim), ``"frame"`` (individual frames),
    or ``"frame_pair"`` (consecutive frame pairs).  The Evaluator uses
    this to calibrate and chunk correctly.
    """

    # --- Declarative metadata (override in subclass) ---
    name: str = ""
    requires_reference: bool = True
    higher_is_better: bool = True
    dependencies: list[str] = []
    needs_gpu: bool = False
    backbone: str | None = None
    batch_unit: str = "video"  # "video", "frame", or "frame_pair"

    def __init__(self) -> None:
        self._device: torch.device = torch.device("cpu")
        self._chunk_size: int | None = None  # set by Evaluator.calibrate()

    # --- GPU lifecycle ---

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: str | torch.device) -> BaseMetric:
        """Move metric (and its internal models) to *device*."""
        self._device = torch.device(device)
        return self

    # --- Setup hook ---

    def setup(self) -> None:
        """Eagerly load models. Called once by :func:`create_evaluator`."""
        pass

    # --- Calibration support ---

    def trial_forward(self, batch_size: int, *, height: int, width: int, num_frames: int) -> None:
        """Run a trial forward pass with *batch_size* batch units.

        Used by :meth:`Evaluator.calibrate` to measure GPU memory per
        batch unit.  Override in subclasses whose ``batch_unit`` is not
        ``"video"`` (e.g. ``"frame"`` or ``"frame_pair"``).

        The default creates *batch_size* dummy videos and calls :meth:`compute`.
        """
        sample = {
            "video": torch.randn(batch_size, num_frames, 3, height, width, device=self.device),
            "reference": torch.randn(batch_size, num_frames, 3, height, width, device=self.device),
            "text_prompt": ["calibration"] * batch_size,
            "audio": ["calibration.wav"] * batch_size,
        }
        with torch.no_grad():
            self.compute(sample)

    # --- Core interface ---

    def _skip(self, sample: dict, reason: str) -> list[MetricResult]:
        """Return a list of B skipped results (score=None)."""
        B = sample["video"].shape[0] if "video" in sample else 1
        return [MetricResult(name=self.name, score=None,
                             details={"skipped": reason}) for _ in range(B)]

    @abstractmethod
    def compute(self, sample: dict) -> list[MetricResult]:
        """Compute the metric on a batched sample.

        ``sample["video"]`` is ``(B, T, C, H, W)``.
        Returns a list of *B* :class:`MetricResult` objects.

        If required inputs (e.g. ``auxiliary_info``, ``text_prompt``)
        are missing, return skipped results via ``self._skip(sample, reason)``
        instead of raising.
        """
        ...
