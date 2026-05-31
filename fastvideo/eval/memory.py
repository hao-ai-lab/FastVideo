"""Small GPU-memory helper used by :class:`EvalWorker`."""

from __future__ import annotations

import gc

import torch


def clear_cache() -> None:
    """Free GPU cache + run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
