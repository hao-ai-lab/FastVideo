# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fastvideo.pipelines import TrainingBatch


class DistillAdapter(ABC):

    @abstractmethod
    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> TrainingBatch:
        raise NotImplementedError
