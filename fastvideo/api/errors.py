# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


class DocumentValidationError(ValueError):
    """Validation error that keeps track of the nested document path."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.path:
            return f"{self.path}: {self.message}"
        return self.message

