# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Flash-MH-MoE forward — fused Triton kernel with deterministic scatter."""

from __future__ import annotations

import os

import torch


def _is_deterministic() -> bool:
    return os.environ.get("MAGI2_DETERMINISTIC", "0") == "1"


def flash_mh_moe_fwd(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """Flash-MH-MoE forward: gather → gate/up GEMM → SwiGLU7 → down GEMM → scatter.

    When ``MAGI2_DETERMINISTIC=1``, uses sequential scatter for bit-exact reproducibility.
    """
    from .triton import mh_moe_fwd_func

    return mh_moe_fwd_func(
        x=x,
        gather_ids=gather_ids,
        probs=probs,
        expert_offsets=expert_offsets,
        W_gate=W_gate,
        W_up=W_up,
        W_down=W_down,
        deterministic=_is_deterministic(),
    )
