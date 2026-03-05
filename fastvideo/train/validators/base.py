# SPDX-License-Identifier: Apache-2.0
"""Deprecated — use fastvideo.train.callbacks instead.

Kept as import shims for backwards compatibility.
"""

from __future__ import annotations

import warnings

from fastvideo.train.callbacks.callback import Callback

warnings.warn(
    "fastvideo.train.validators.base is deprecated. "
    "Use fastvideo.train.callbacks instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Provide the old names as aliases so that existing
# imports do not break immediately.
Validator = Callback
ValidationRequest = None  # type: ignore[assignment]
