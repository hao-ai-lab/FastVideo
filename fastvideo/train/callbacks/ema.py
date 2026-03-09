# SPDX-License-Identifier: Apache-2.0
"""EMA (Exponential Moving Average) callback.

Owns the full EMA lifecycle: creation, per-step updates, weight
swapping for validation, and checkpoint state.  All EMA config
lives under ``callbacks.ema`` in the YAML file.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.training.training_utils import EMA_FSDP

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class EMACallback(Callback):
    """Manage EMA shadow weights for the student transformer.

    All configuration lives in the YAML ``callbacks.ema`` section:

    .. code-block:: yaml

        callbacks:
          ema:
            decay: 0.9999
            start_iter: 0

    The callback creates an ``EMA_FSDP`` instance at train start,
    updates it after each optimizer step, and exposes an
    ``ema_context()`` context manager for temporarily swapping
    EMA weights into the live model (used by validation).
    """

    def __init__(
        self,
        *,
        decay: float = 0.9999,
        start_iter: int = 0,
    ) -> None:
        self._decay = float(decay)
        self._start_iter = int(start_iter)
        self._ema_started = False
        self.student_ema: EMA_FSDP | None = None

    # ----------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        student = getattr(method, "student", None)
        if student is None or student.transformer is None:
            raise ValueError("No student transformer found on method, cannot initialize EMA")

        logger.info(
            "Initializing EMA (local_shard) with "
            "decay=%s from student transformer",
            self._decay,
        )
        self.student_ema = EMA_FSDP(
            student.transformer,
            decay=self._decay,
            mode="local_shard",
        )
        logger.info(
            "EMA callback enabled (decay=%s, "
            "start_iter=%d).",
            self._decay,
            self._start_iter,
        )

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        if self.student_ema is None:
            return

        if iteration < self._start_iter:
            return
        if not self._ema_started:
            logger.info(
                "Starting EMA updates at iteration %d "
                "(re-initializing shadow from current "
                "model).",
                iteration,
            )
            self.student_ema._init_shadow(method.student.transformer, )
            self._ema_started = True

        self.student_ema.update(method.student.transformer, )

        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            tracker.log(
                {"ema/decay": self.student_ema.decay},
                iteration,
            )

    # ----------------------------------------------------------
    # EMA context manager
    # ----------------------------------------------------------

    @contextlib.contextmanager
    def ema_context(
        self,
        transformer: torch.nn.Module,
    ) -> Generator[torch.nn.Module, None, None]:
        """Temporarily swap EMA weights into *transformer*.

        If EMA is not active, yields the transformer unchanged.
        """
        if (self.student_ema is not None and self._ema_started):
            with self.student_ema.apply_to_model(transformer, ):
                yield transformer
        else:
            yield transformer

    # ----------------------------------------------------------
    # Checkpoint state
    # ----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        if self.student_ema is None:
            return {}
        return {
            "student_ema": self.student_ema.state_dict(),
            "ema_started": self._ema_started,
        }

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        ema_state = state_dict.get("student_ema")
        if (ema_state is not None and self.student_ema is not None):
            self.student_ema.load_state_dict(ema_state)
        self._ema_started = bool(state_dict.get("ema_started", False), )
