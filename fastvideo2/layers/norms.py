"""Norms with explicit fp32 compute — extracted verbatim from the vendored
official Wan2.1 model (wan/modules/model.py @ 9737cba, Apache-2.0).
Bitwise equivalence to the official implementation is enforced by the anchor
gate; do not reorder casts.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm computed in fp32, cast back, THEN multiplied by ``weight``.

    The multiply runs in the *weight's* dtype: with fp32 weight storage under
    bf16 autocast this deliberately PROMOTES activations back to fp32 — which
    is why official's attention receives fp32-promoted q/k and casts them
    itself. Load-bearing for official parity; do not "simplify" the cast order.
    (Official name: WanRMSNorm. Parameter key: ``weight``.)
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32 and cast back to the input dtype.
    (Official name: WanLayerNorm.)"""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)
