"""Small GPU-memory helpers used by :class:`Evaluator`."""

from __future__ import annotations

import gc

import torch


def is_batch_too_large(exc: Exception) -> bool:
    """True if *exc* indicates the batch should be reduced (OOM, 32-bit
    indexing limits, allocator failures)."""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return any(s in msg for s in (
        "out of memory",
        "CUDNN_STATUS_NOT_SUPPORTED",
        "DefaultCPUAllocator: can't allocate memory",
        "canUse32BitIndexMath",
    ))


def clear_cache() -> None:
    """Free GPU cache + run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def slice_sample(sample: dict, start: int, end: int) -> dict:
    """Slice every batched value in *sample* along the leading B dim."""
    out = {}
    for k, v in sample.items():
        if (isinstance(v, torch.Tensor) and v.dim() >= 2) or isinstance(v, list):
            out[k] = v[start:end]
        else:
            out[k] = v
    return out
