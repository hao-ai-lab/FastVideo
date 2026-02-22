# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod


class DistillValidator(ABC):
    @abstractmethod
    def log_validation(self, step: int) -> None:
        raise NotImplementedError
