# SPDX-License-Identifier: Apache-2.0
"""Deprecated — use fastvideo.train.callbacks.ValidationCallback.

Kept as an import shim for backwards compatibility.
"""

from __future__ import annotations

import warnings

from fastvideo.train.callbacks.validation import (
    ValidationCallback,
)

warnings.warn(
    "fastvideo.train.validators.wangame is deprecated. "
    "Use fastvideo.train.callbacks.ValidationCallback "
    "instead.",
    DeprecationWarning,
    stacklevel=2,
)

WanGameValidator = ValidationCallback
