# SPDX-License-Identifier: Apache-2.0
"""Gradient norm clipping callback.

Clips gradients on modules returned by
``method.get_grad_clip_targets()`` before the optimizer step.
Optionally logs per-module grad norms to the tracker.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.utils.optimizer import (
    clip_grad_norm_if_needed, )

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
    ) -> None:
        self._max_grad_norm = float(max_grad_norm)
        self._log_grad_norms = bool(log_grad_norms)

    def on_before_optimizer_step(
        self,
        method: TrainingMethod,
        iteration: int = 0,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        max_norm = self._max_grad_norm
        if max_norm <= 0.0:
            return

        targets = method.get_grad_clip_targets(iteration)
        tracker = getattr(method, "tracker", None)

        for name, module in targets.items():
            grad_norm = clip_grad_norm_if_needed(
                module,
                max_norm,
            )
            if (self._log_grad_norms and tracker is not None and grad_norm > 0.0):
                tracker.log(
                    {f"grad_norm/{name}": grad_norm},
                    iteration,
                )
