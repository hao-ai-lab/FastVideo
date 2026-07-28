# SPDX-License-Identifier: Apache-2.0
"""Staged unfreeze for the WanTrack track pathway.

Tests whether a RANDOM-init track encoder can be bootstrapped directly on stage-1 data, which
would remove the need for the separate overfit -> merge stages. For the first ``warmup_steps``
optimizer steps only the track pathway learns; the pretrained DiT is held still. After that the
DiT joins at its normal LR, either instantly or ramped in over ``ramp_steps``.

Requires ``WANTRACK_TRACK_GROUP=1`` so the optimizer has the named 'track'/'base' param groups
(see FineTuneMethod._build_track_param_groups).

Why lr=0 rather than requires_grad=False: params are bound to the optimizer at construction, so
flipping requires_grad later would leave the DiT permanently outside the optimizer. Setting the
group's lr to 0 is a true freeze for AdamW — the update is scaled by lr AND so is the decoupled
weight decay (p *= 1 - lr*wd), so at lr=0 the weights genuinely do not move. Zeroing grads
instead would still let weight decay shrink them.

The LR must be (re)written every step from ``on_before_optimizer_step``, because the LR
scheduler rewrites ``group['lr']`` on its own step; this hook runs after that and immediately
before ``optimizer.step()``, so it wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class TrackWarmupCallback(Callback):
    """Hold the pretrained DiT at lr=0 while the track pathway warms up."""

    def __init__(
        self,
        *,
        warmup_steps: int = 500,
        ramp_steps: int = 0,
        base_group: str = "base",
        track_group: str = "track",
        log_every: int = 50,
    ) -> None:
        self._warmup = int(warmup_steps)
        self._ramp = max(int(ramp_steps), 0)
        self._base_name = str(base_group)
        self._track_name = str(track_group)
        self._log_every = max(int(log_every), 1)
        self._checked = False

    def _scale_for(self, iteration: int) -> float:
        """0 during warmup, then linearly to 1 over ramp_steps (instant if ramp_steps=0)."""
        if iteration < self._warmup:
            return 0.0
        if self._ramp <= 0:
            return 1.0
        return min(1.0, (iteration - self._warmup) / float(self._ramp))

    def on_before_optimizer_step(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        scale = self._scale_for(iteration)
        applied = False
        for opt in method.get_optimizers(iteration):
            for group in opt.param_groups:
                name = group.get("name")
                if name == self._base_name:
                    group["lr"] = group["lr"] * scale
                    applied = True

        if not self._checked:
            self._checked = True
            if not applied:
                # Fail loud: silently running a plain finetune would look like a successful
                # experiment while testing nothing.
                raise RuntimeError(
                    f"TrackWarmupCallback found no param group named {self._base_name!r}. "
                    "Set WANTRACK_TRACK_GROUP=1 so the optimizer is built with named "
                    "'track'/'base' groups.")
            logger.info(
                "[WANTRACK] TrackWarmupCallback active: DiT frozen (lr x0) for %d steps, "
                "then ramped over %d steps", self._warmup, self._ramp)

        if iteration % self._log_every == 0:
            tracker = getattr(method, "tracker", None)
            if tracker is not None:
                lrs = {}
                for opt in method.get_optimizers(iteration):
                    for group in opt.param_groups:
                        if group.get("name"):
                            lrs[f"lr/{group['name']}"] = float(group["lr"])
                lrs["lr/base_scale"] = scale
                if lrs:
                    tracker.log(lrs, iteration)
