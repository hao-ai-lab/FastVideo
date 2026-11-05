# SPDX-License-Identifier: Apache-2.0
"""Exact Q/K preconditioning ahead of NVFP4 (per-16 block) Q/K quantization.

Both transforms leave the attention output unchanged in exact arithmetic and only redistribute what the FP4 blocks
have to represent:

* ``smooth_k``: subtract each head's token-mean of K. ``q . k_mean`` is the same for every key of a query row, and
  softmax is shift-invariant per row.
* ``hadamard``: rotate Q and K by the same orthonormal head-dim Hadamard matrix ``H`` (``Q H (K H)^T = Q K^T``),
  which spreads outlier channels across the 16-wide scale blocks.

The QAD emulator and the deployed FlashAttention-4 NVFP4 path apply the same function, so training and inference
quantize identical tensors.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch

QK_TRANSFORMS = {
    "none": (False, False),
    "smooth_k": (True, False),
    "hadamard": (False, True),
    "smooth_k+hadamard": (True, True),
}


def parse_qk_transform(name: str | None) -> tuple[bool, bool]:
    """``"smooth_k+hadamard"`` -> ``(smooth_k, hadamard)``; ``None``/``"none"`` -> no transform."""
    key = (name or "none").strip().lower()
    if key not in QK_TRANSFORMS:
        raise ValueError(f"unknown NVFP4 Q/K transform {name!r}; expected one of {', '.join(QK_TRANSFORMS)}")
    return QK_TRANSFORMS[key]


def parse_block_indices(raw: str | None) -> frozenset[int]:
    """``"45, 49"`` -> ``frozenset({45, 49})``; empty/``None`` -> empty set."""
    items = [item.strip() for item in (raw or "").split(",") if item.strip()]
    try:
        indices = frozenset(int(item) for item in items)
    except ValueError as error:
        raise ValueError(f"expected a comma-separated list of block indices, got {raw!r}") from error
    if any(index < 0 for index in indices):
        raise ValueError(f"block indices must be non-negative, got {raw!r}")
    return indices


@lru_cache(maxsize=8)
def _hadamard(n: int, device: torch.device) -> torch.Tensor:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard rotation needs a power-of-two head dim, got {n}")
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(device)


def hadamard_matrix(n: int, device: torch.device | str) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix of size ``n`` (fp32)."""
    return _hadamard(n, torch.device(device))


def precondition_qk(query: torch.Tensor, key: torch.Tensor, *, smooth_k: bool,
                    hadamard: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform ``[B, L, H, D]`` Q/K (computed in fp32, returned in the input dtype)."""
    if not (smooth_k or hadamard):
        return query, key
    q, k = query.float(), key.float()
    if smooth_k:
        k = k - k.mean(dim=1, keepdim=True)
    if hadamard:
        rotation = hadamard_matrix(q.shape[-1], q.device)
        q, k = q @ rotation, k @ rotation
    return q.to(query.dtype), k.to(key.dtype)


__all__ = ["QK_TRANSFORMS", "hadamard_matrix", "parse_block_indices", "parse_qk_transform", "precondition_qk"]
