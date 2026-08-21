# SPDX-License-Identifier: Apache-2.0
"""Inference-only MiniMax H3 fusions adapted from NVlabs/Sana Sol-Engine.

Source: https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/GB200
"""

from .modulation import (
    fused_residual_gate_rmsnorm_modulate,
    fused_rmsnorm_modulate,
)
from .qknorm_rope import HAVE_TRITON, fused_qknorm_rope
from .swiglu import minimax_h3_swiglu

__all__ = [
    "HAVE_TRITON",
    "fused_qknorm_rope",
    "fused_residual_gate_rmsnorm_modulate",
    "fused_rmsnorm_modulate",
    "minimax_h3_swiglu",
]
