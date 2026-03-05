# SPDX-License-Identifier: Apache-2.0
"""EMA (Exponential Moving Average) callback.

Updates EMA shadow weights after each training step.  The model owns
the EMA network (created by ``ModelBase._setup_ema``); this callback
only performs the ``lerp_`` update.
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

    The EMA network lives on the method (``method.ema``).
    If the method was created with ``use_ema: false``, the callback
    detects this at train start and disables itself gracefully.

    Supports three beta strategies:
    - ``constant``: fixed ``beta`` every step.
    - ``power``:  ``(1 - 1/t)^(gamma+1)``.
    - ``halflife``: half-life in k-images with optional ramp-up.
    """

    def __init__(
        self,
        *,
        type: str = "constant",
        beta: float = 0.9999,
        gamma: float = 16.97,
        ema_halflife_kimg: float = 500.0,
        ema_rampup_ratio: float | None = 0.05,
        start_iter: int = 0,
        batch_size: int = 1,
    ) -> None:
        self._type = str(type)
        self._beta = float(beta)
        self._gamma = float(gamma)
        self._ema_halflife_kimg = float(ema_halflife_kimg)
        self._ema_rampup_ratio = (
            float(ema_rampup_ratio)
            if ema_rampup_ratio is not None
            else None
        )
        self._start_iter = int(start_iter)
        self._batch_size = int(batch_size)
        self._enabled = True

    # ----------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        ema = getattr(method, "ema", None)
        if ema is None:
            self._enabled = False
            logger.info(
                "EMA not found on method; "
                "EMA callback disabled.",
            )
            return

        assert not ema.training, (
            "EMA should be in eval mode"
        )
        for name, p in ema.named_parameters():
            assert not p.requires_grad, (
                f"EMA parameter {name} should not "
                f"require gradients"
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
                "Starting EMA %r updates at iteration %d.",
                "ema",
                iteration,
            )

        beta = self._compute_beta(iteration)
        ema = method.ema
        ema_state = ema.state_dict()

        with torch.no_grad():
            for name, p_net in (
                method.student.transformer.named_parameters()
            ):
                full = self._gather_full(p_net)
                ema_key = name.replace(
                    "_checkpoint_wrapped_module.", "",
                )
                if ema_key not in ema_state:
                    if iteration == self._start_iter:
                        logger.warning(
                            "EMA param %r not found, "
                            "skipping.",
                            ema_key,
                        )
                    continue
                ema_p = ema_state[ema_key]
                val = full.to(
                    device=ema_p.device,
                    dtype=ema_p.dtype,
                )
                if iteration == self._start_iter:
                    ema_p.copy_(val)
                else:
                    ema_p.lerp_(val, 1.0 - beta)

            for name, buf in (
                method.student.transformer.named_buffers()
            ):
                if name in ema_state:
                    ema_state[name].copy_(
                        buf.to(
                            device=ema_state[name].device,
                            dtype=ema_state[name].dtype,
                        )
                    )

        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            tracker.log(
                {"ema/beta": beta},
                iteration,
            )

    # ----------------------------------------------------------
    # Beta strategies
    # ----------------------------------------------------------

    def _compute_beta(self, iteration: int) -> float:
        if self._type == "constant":
            return self._beta
        if self._type == "power":
            it = max(iteration, 1)
            return (1.0 - 1.0 / it) ** (self._gamma + 1)
        if self._type == "halflife":
            return self._halflife_beta(iteration)
        raise ValueError(
            f"Invalid EMA type: {self._type!r}"
        )

    def _halflife_beta(self, iteration: int) -> float:
        hl_nimg = self._ema_halflife_kimg * 1000.0
        cur_nimg = iteration * self._batch_size
        if self._ema_rampup_ratio is not None:
            hl_nimg = min(
                hl_nimg,
                cur_nimg * self._ema_rampup_ratio,
            )
        return 0.5 ** (
            self._batch_size / max(hl_nimg, 1e-8)
        )

    # ----------------------------------------------------------
    # FSDP helper
    # ----------------------------------------------------------

    @staticmethod
    def _gather_full(
        param: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(param, "full_tensor"):
            if param.device.type == "cpu":
                return param.to("cuda").full_tensor()
            return param.full_tensor()
        return param
