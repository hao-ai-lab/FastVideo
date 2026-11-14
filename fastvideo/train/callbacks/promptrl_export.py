# SPDX-License-Identifier: Apache-2.0
"""PromptRL inference-bundle export callback.

Exports a PromptRL bundle (manifest + refiner PEFT adapter + generator
LoRA safetensors) from the live roles at train end, and optionally at a
step cadence for long runs.  Only rank 0 writes; LoRA weights are
replicated across ranks so the rank-0 copy is complete.
"""

from __future__ import annotations

import os
from typing import Any

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

logger = init_logger(__name__)


class PromptRLBundleExportCallback(Callback):
    """Write PromptRL inference bundles from a PromptRLMethod."""

    def __init__(
        self,
        *,
        output_dir: str,
        every_steps: int = 0,
    ) -> None:
        if not output_dir:
            raise ValueError("callbacks.promptrl_export.output_dir must be set")
        self._output_dir = str(output_dir)
        self._every_steps = max(0, int(every_steps))

    # ------------------------------------------------------------------

    def _is_rank0(self) -> bool:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def _export(self, method: Any, *, suffix: str = "") -> None:
        export = getattr(method, "export_bundle", None)
        if export is None:
            logger.warning("promptrl_export callback attached to %s which has no "
                           "export_bundle(); skipping", type(method).__name__)
            return
        if not self._is_rank0():
            return
        target = os.path.join(self._output_dir + suffix) if suffix else self._output_dir
        written = export(target)
        logger.info("Exported PromptRL bundle to %s (%s)", target, sorted(written))

    # ------------------------------------------------------------------

    def on_training_step_end(self, method: Any, loss_dict: dict[str, Any], iteration: int = 0) -> None:
        if self._every_steps <= 0 or iteration <= 0 or iteration % self._every_steps != 0:
            return
        self._export(method, suffix=f"-step-{iteration}")

    def on_train_end(self, method: Any, iteration: int = 0) -> None:
        self._export(method)
