"""Compatibility shim for the legacy non-QAT Triton attention import path.

Historically callers imported
``fastvideo_kernel.triton_kernels.fused_attention`` directly. The shared
implementation now lives in ``attn_qat_train.py`` and is parameterized by the
``IS_QAT`` flag. This module preserves the original public API for tests and
downstream users while always dispatching to the non-QAT configuration.
"""

from __future__ import annotations

import torch

from .attn_qat_train import attention as _attention


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
    warp_specialize: bool = True,
) -> torch.Tensor:
    """Run the shared Triton attention kernel in non-QAT mode."""
    use_qat_qkv_backward = True
    smooth_k = False
    is_qat = False
    two_level_quant_p = False
    fake_quant_p = False
    use_high_prec_o = False
    smooth_q = False
    use_global_sf_p = False
    use_global_sf_qkv = False

    return _attention(
        q,
        k,
        v,
        causal,
        sm_scale,
        use_qat_qkv_backward,
        smooth_k,
        warp_specialize,
        is_qat,
        two_level_quant_p,
        fake_quant_p,
        use_high_prec_o,
        smooth_q,
        use_global_sf_p,
        use_global_sf_qkv,
    )


__all__ = ["attention"]
