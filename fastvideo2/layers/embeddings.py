"""Timestep embeddings — extracted verbatim from the vendored official Wan2.1
model (@ 9737cba). The fp64 position math is part of the model's numerics.
"""
from __future__ import annotations

import torch


def sinusoidal_embedding_1d(dim: int, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x
