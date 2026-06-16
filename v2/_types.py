"""Shared lightweight type aliases for v2.

The core (card/loop/runtime/cache/parity/program/request/training) is numpy-only
and CPU-testable. Tensors are therefore typed structurally as ``TensorLike``:
on the CPU test path they are ``numpy.ndarray``; on a GPU box the same code runs
with ``torch.Tensor``. The core never imports torch — only model-component
*adapters* do, lazily (see ``v2/card/components.py``).
"""
from __future__ import annotations

from typing import Any

# A tensor-like object. numpy.ndarray (CPU tests) or torch.Tensor (GPU). The core
# only relies on duck-typed ops provided by the active backend (see card/backend).
TensorLike = Any

Shape = tuple[int, ...]

# A stable content hash (hex) used for cache keys and provenance.
Hash = str
