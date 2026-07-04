"""Shared lightweight type aliases for v2.

The core (card/loop/runtime/cache/parity/program/request) is numpy-only
and CPU-testable, so tensors are typed structurally as ``TensorLike``: a
``numpy.ndarray`` on the CPU test path, a ``torch.Tensor`` on GPU. The core never
imports torch — only model-component adapters do, lazily (see
``v2/card/components.py``).
"""
from __future__ import annotations

from typing import Any

# A tensor-like object. numpy.ndarray (CPU tests) or torch.Tensor (GPU). The core
# only relies on duck-typed ops provided by the active backend (see card/backend).
TensorLike = Any

Shape = tuple[int, ...]

# A stable content hash (hex) used for cache keys and provenance.
Hash = str
