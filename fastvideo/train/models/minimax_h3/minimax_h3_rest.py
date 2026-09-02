# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 adapter for immutable REST/AMD teacher-trajectory caches."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from fastvideo.dataset.h3_rest_cache import (
    H3RESTCacheDataset,
    build_h3_rest_cache_dataloader,
)
from fastvideo.distributed import get_sp_group
from fastvideo.train.models.minimax_h3.minimax_h3_rvm import MiniMaxH3RVMModel


class MiniMaxH3RESTModel(MiniMaxH3RVMModel):
    """Trainable FastH3 LoRA fed by frozen, scored full-H3 trajectories.

    ``training.data.data_path`` points at one completed REST cache. The model
    deliberately does not construct the ordinary parquet dataloader: every
    optimizer step consumes a cached H3 trajectory segment and its immutable
    offline reward metadata.
    """

    def __init__(
        self,
        *,
        student_timesteps: Sequence[int | float],
        verify_cache_hashes: bool = False,
        **kwargs: Any,
    ) -> None:
        self._rest_student_timesteps = self._validate_student_timesteps(
            student_timesteps
        )
        self._rest_verify_cache_hashes = bool(verify_cache_hashes)
        self.rest_cache_dataset: H3RESTCacheDataset | None = None
        super().__init__(**kwargs)

    @staticmethod
    def _validate_student_timesteps(
        values: Sequence[int | float],
    ) -> tuple[float, ...]:
        timesteps = tuple(float(value) for value in values)
        if len(timesteps) < 2:
            raise ValueError("models.student.student_timesteps needs at least two values")
        if any(not math.isfinite(value) for value in timesteps):
            raise ValueError("models.student.student_timesteps must be finite")
        if any(
            left <= right
            for left, right in zip(timesteps[:-1], timesteps[1:], strict=True)
        ):
            raise ValueError(
                "models.student.student_timesteps must be strictly descending, "
                f"got {list(timesteps)}"
            )
        return timesteps

    @property
    def rest_student_timesteps(self) -> tuple[float, ...]:
        return self._rest_student_timesteps

    @property
    def rest_cache_fingerprint(self) -> str:
        if self.rest_cache_dataset is None:
            raise RuntimeError("REST cache is not initialized")
        return self.rest_cache_dataset.summary.fingerprint

    def init_preprocessors(self, training_config: Any) -> None:
        cache_path = training_config.data.data_path
        if not isinstance(cache_path, str) or not cache_path.strip():
            raise ValueError(
                "H3 REST requires training.data.data_path to be one cache directory"
            )
        self.sp_group = get_sp_group()
        dataset, dataloader = build_h3_rest_cache_dataloader(
            cache_path,
            batch_size=int(training_config.data.train_batch_size),
            num_data_workers=int(training_config.data.dataloader_num_workers),
            seed=int(training_config.data.seed),
            verify_file_hashes=self._rest_verify_cache_hashes,
            expected_student_timesteps=self._rest_student_timesteps,
        )
        self.rest_cache_dataset = dataset
        self.dataloader = dataloader
        self.start_step = 0


__all__ = ["MiniMaxH3RESTModel"]
