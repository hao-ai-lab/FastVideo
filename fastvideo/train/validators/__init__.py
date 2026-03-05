# SPDX-License-Identifier: Apache-2.0
"""Deprecated — use fastvideo.train.callbacks instead."""

from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.callbacks.validation import (
    ValidationCallback,
)

# Backwards-compatible aliases.
Validator = Callback
WanValidator = ValidationCallback

__all__ = [
    "Validator",
    "WanValidator",
    "ValidationCallback",
]
