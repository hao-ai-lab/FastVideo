"""Attn-QAT training attention — vendored from fastvideo-main
``fastvideo/attention/backends/attn_qat_train.py`` (@ the goldens manifest
commit). Calls the SAME ``fastvideo_kernel.triton_kernels.attn_qat_train``
Triton kernel (fake-quantized SageAttention-style forward with quantized
backward, straight-through). head_dim 128 only; fails closed without the
kernel package (a dense fallback would train a different model).

Flag set is main's verbatim: is_qat, fake_quant_p_bwd, use_qat_qkv_backward,
use_high_prec_o on; warp_specialize disabled on Blackwell (sm100/sm120 —
Triton NVWS pass aborts there); sm_scale = d**-0.5.

Import stays torch-free.
"""
from __future__ import annotations

from typing import Any


def attn_qat_train(q_blhd: Any, k_blhd: Any, v_blhd: Any,
                   is_causal: bool = False, sm_scale: float | None = None) -> Any:
    import torch
    try:
        from fastvideo_kernel.triton_kernels.attn_qat_train import attention
    except ImportError as e:  # fail closed
        raise RuntimeError(
            "ATTN_QAT_TRAIN requires the fastvideo_kernel package (triton "
            "attn_qat_train); refusing to substitute dense attention") from e

    q = q_blhd.permute(0, 2, 1, 3).contiguous()
    k = k_blhd.permute(0, 2, 1, 3).contiguous()
    v = v_blhd.permute(0, 2, 1, 3).contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    warp_specialize = torch.cuda.get_device_capability()[0] not in (10, 12)
    o = attention(
        q, k, v, is_causal, sm_scale,
        True,   # use_qat_qkv_backward
        False,  # smooth_k
        warp_specialize,
        True,   # is_qat
        False,  # two_level_quant_p_sage3
        True,   # fake_quant_p_bwd
        True,   # use_high_prec_o
        False,  # smooth_q
        False,  # use_global_sf_p
        False,  # use_global_sf_qkv
    )
    return o.permute(0, 2, 1, 3).contiguous()
