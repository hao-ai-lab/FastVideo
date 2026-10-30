# SPDX-License-Identifier: Apache-2.0
"""MotionKernel H3 kernels (opt-in through ``FASTVIDEO_MINIMAX_H3_FUSIONS=mk``).

These are the MotionKernel agent-discovered Triton kernels for the MiniMax-H3
inference blocks, kept separate from the Sol-Engine kernels in the parent
package so the default path stays byte-for-byte identical. Each kernel is a
drop-in replacement for one Sol fusion or eager expression:

============================================  ======================================
MotionKernel                                   replaces
============================================  ======================================
``qkv_prologue``                              2x ``fused_qknorm_rope`` + the
                                              ``torch.cat`` QKV stack
``rmsnorm_modulate``                          ``fused_rmsnorm_modulate``
``residual_gate_rmsnorm_modulate``            ``fused_residual_gate_rmsnorm_modulate``
``swiglu``                                    ``minimax_h3_swiglu``
``gated_residual``                            the compiled ``_gated_residual``
============================================  ======================================

All five run in fp32 internally with a single rounding at the bf16/fp16 store
and require the production H3 norm eps of ``1e-5`` (enforced at the call sites).
"""

try:
    import triton  # noqa: F401  (each module imports it directly)

    from .gated_residual import gated_residual
    from .qkv_prologue import qkv_prologue
    from .residual_gate_rmsnorm_modulate import residual_gate_rmsnorm_modulate
    from .rmsnorm_modulate import rmsnorm_modulate
    from .swiglu import swiglu

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - environments without Triton
    gated_residual = None
    qkv_prologue = None
    residual_gate_rmsnorm_modulate = None
    rmsnorm_modulate = None
    swiglu = None
    HAVE_TRITON = False

__all__ = [
    "HAVE_TRITON",
    "gated_residual",
    "qkv_prologue",
    "residual_gate_rmsnorm_modulate",
    "rmsnorm_modulate",
    "swiglu",
]
