# SPDX-License-Identifier: Apache-2.0
"""
Engine hook implementations for denoising runs.
"""

from __future__ import annotations

import time

from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
from fastvideo.pipelines.stages.denoising_strategies import StrategyState

logger = init_logger(__name__)


class PerfLoggingHook:
    """
    Record per-step and total denoising times into batch logging info.
    """

    def __init__(self, stage_name: str = "DenoisingEngine") -> None:
        self.stage_name = stage_name
        self._batch: ForwardBatch | None = None
        self._run_start = 0.0
        self._step_starts: dict[int, float] = {}
        self._step_times_ms: list[float] = []

    def on_init(self, engine: DenoisingEngine, batch: ForwardBatch, args):
        self._batch = batch

    def pre_run(self, state: StrategyState) -> None:
        self._run_start = time.perf_counter()
        self._step_starts.clear()
        self._step_times_ms.clear()

    def pre_step(self, state: StrategyState, step_idx: int, t) -> None:
        self._step_starts[step_idx] = time.perf_counter()

    def post_step(self, state: StrategyState, step_idx: int, t) -> None:
        start = self._step_starts.pop(step_idx, None)
        if start is None:
            return
        self._step_times_ms.append((time.perf_counter() - start) * 1000.0)

    def post_run(self, state: StrategyState, batch: ForwardBatch) -> None:
        total_ms = (time.perf_counter() - self._run_start) * 1000.0
        target_batch = self._batch or batch
        if target_batch is None:
            return
        target_batch.logging_info.add_stage_metric(self.stage_name,
                                                   "denoise_step_times_ms",
                                                   list(self._step_times_ms))
        target_batch.logging_info.add_stage_metric(self.stage_name,
                                                   "denoise_total_ms", total_ms)
        if not self._step_times_ms:
            return
        if not self._is_primary_rank():
            return
        mean_ms = sum(self._step_times_ms) / len(self._step_times_ms)
        logger.info(
            "[%s] denoise steps=%d total_ms=%.2f mean_step_ms=%.2f min=%.2f max=%.2f",
            self.stage_name,
            len(self._step_times_ms),
            total_ms,
            mean_ms,
            min(self._step_times_ms),
            max(self._step_times_ms),
        )

    def _is_primary_rank(self) -> bool:
        try:
            from fastvideo.distributed import get_world_group
            return get_world_group().local_rank == 0
        except Exception:
            return True
