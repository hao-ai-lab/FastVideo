# SPDX-License-Identifier: Apache-2.0
"""Gradient sanitation and norm clipping callback.

Sanitizes gradients on modules returned by
``method.get_grad_sanitize_targets()`` and clips modules returned by
``method.get_grad_clip_targets()`` before the optimizer step.
Optionally logs per-module grad norms to the tracker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.utils.lora import sync_replicated_lora_gradients
from fastvideo.train.utils.optimizer import (
    clip_grad_norm_if_needed,
    measure_grad_norm,
    sanitize_non_finite_gradients,
)

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class GradNormClipCallback(Callback):
    """Clip gradient norms before the optimizer step.

    ``max_grad_norm`` must be set explicitly in the callback
    config (``callbacks.grad_clip.max_grad_norm``).
    """

    def __init__(
        self,
        *,
        max_grad_norm: float = 1.0,
        log_grad_norms: bool = True,
        measure_unclipped_grad_norms: bool = False,
        sanitize_non_finite: bool = False,
    ) -> None:
        self._sanitize_non_finite = bool(sanitize_non_finite)
        self._max_grad_norm = float(max_grad_norm)
        self._log_grad_norms = bool(log_grad_norms)
        self._measure_unclipped_grad_norms = bool(measure_unclipped_grad_norms)
        self.last_grad_norms: dict[str, float] = {}
        self.last_non_finite_grad_counts: dict[str, int] = {}

    def on_before_optimizer_step(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        max_norm = self._max_grad_norm
        tracker = getattr(method, "tracker", None)
        self.last_grad_norms = {}
        self.last_non_finite_grad_counts = {}

        sanitize_getter = getattr(method, "get_grad_sanitize_targets", method.get_grad_clip_targets)
        sanitize_targets = sanitize_getter(iteration)
        for name, module in sanitize_targets.items():
            sync_replicated_lora_gradients(module)
            count = sanitize_non_finite_gradients(module) if self._sanitize_non_finite else 0
            self.last_non_finite_grad_counts[name] = count
            if count > 0:
                logger.warning("Sanitized %d non-finite gradient values in %s", count, name)
                if tracker is not None:
                    tracker.log(
                        {f"non_finite_gradients/{name}": count},
                        iteration,
                    )

        clip_targets = method.get_grad_clip_targets(iteration) if max_norm > 0.0 else {}
        norm_targets = dict(clip_targets)
        if self._measure_unclipped_grad_norms:
            for name, module in sanitize_targets.items():
                norm_targets.setdefault(name, module)

        for name, module in norm_targets.items():
            grad_norm = (clip_grad_norm_if_needed(module, max_norm)
                         if name in clip_targets else measure_grad_norm(module))
            self.last_grad_norms[name] = float(grad_norm)
            if (self._log_grad_norms and tracker is not None and grad_norm > 0.0):
                tracker.log(
                    {f"grad_norm/{name}": grad_norm},
                    iteration,
                )
