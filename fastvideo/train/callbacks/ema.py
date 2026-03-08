# SPDX-License-Identifier: Apache-2.0
"""EMA (Exponential Moving Average) callback.

Updates the ``EMA_FSDP`` shadow weights (local FSDP shards on CPU)
after each training step.  The method owns the ``generator_ema``
instance; this callback only calls ``update()``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class EMACallback(Callback):
    """Update EMA parameters after each optimizer step.

    The ``EMA_FSDP`` instance lives on the method
    (``method.generator_ema``).  If the method was created with
    ``use_ema: false``, the callback detects this at train start
    and disables itself gracefully.

    Config knobs (set in method YAML under ``use_ema`` /
    ``ema_decay``):
    - ``ema_start_step``: first step at which EMA begins
      updating (default 0).
    """

    def __init__(
        self,
        *,
        start_iter: int = 0,
    ) -> None:
        self._start_iter = int(start_iter)
        self._enabled = True

    # ----------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        ema = getattr(method, "generator_ema", None)
        if ema is None:
            self._enabled = False
            logger.info(
                "EMA not found on method; "
                "EMA callback disabled.",
            )
            return
        logger.info(
            "EMA callback enabled (decay=%s, "
            "start_iter=%d).",
            ema.decay,
            self._start_iter,
        )

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        if not self._enabled:
            return

        if iteration < self._start_iter:
            return
        if iteration == self._start_iter:
            logger.info(
                "Starting EMA updates at iteration %d.",
                iteration,
            )

        assert method.generator_ema is not None
        method.generator_ema.update(
            method.student.transformer,
        )

        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            tracker.log(
                {"ema/decay": method.generator_ema.decay},
                iteration,
            )
