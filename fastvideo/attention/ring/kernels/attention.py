# SPDX-License-Identifier: Apache-2.0
#
# Adapted from:
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/kernels/attention.py
#
# FastVideo changes:
# - use local Ring Attention capability flags;
# - keep kernel bindings independent of yunchang global process-group state;
# - use FastVideo's public FlashAttention forward API for version tolerance;
# - retain the upstream PyTorch, FA3, AITER, FlashInfer, and NPU adapters.

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ..capabilities import (
    HAS_AITER,
    HAS_FLASH_ATTN,
    HAS_FLASH_ATTN_HOPPER,
    HAS_FLASHINFER,
    HAS_NPU,
)

_scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention
_scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention

# Moore Threads replaces the corresponding ATen FlashAttention operator. The
# import is harmless on ordinary CUDA installations because torch_musa is not
# present there.
try:
    import torch_musa  # noqa: F401

    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_attention_flash_musa
    _scaled_dot_product_efficient_attention = None
except ModuleNotFoundError:
    pass

if HAS_FLASH_ATTN:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_backward,
        _flash_attn_forward,
    )

if HAS_FLASH_ATTN_HOPPER:
    from flash_attn_interface import _flash_attn_backward as flash_attn_func_hopper_backward
    from flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper
    from flash_attn_interface import flash_attn_func as flash3_attn_func
else:
    flash_attn_forward_hopper = None
    flash_attn_func_hopper_backward = None
    flash3_attn_func = None

if HAS_FLASHINFER:
    from flashinfer.prefill import single_prefill_with_kv_cache

    _LOG2_E = math.log2(math.e)

if HAS_AITER:
    from aiter import flash_attn_func as flash_attn_func_aiter

if HAS_NPU:
    import torch_npu


def pytorch_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
    op_type: str = "flash",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a PyTorch SDPA kernel on tensors in ``[B, S, H, D]`` layout."""
    del window_size, softcap, alibi_slopes, return_softmax
    if op_type not in {"flash", "efficient", "math", "cudnn"}:
        raise ValueError(f"Invalid op_type: {op_type}")

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if op_type == "flash":
        out, lse = _scaled_dot_product_flash_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    elif op_type == "efficient":
        if _scaled_dot_product_efficient_attention is None:
            raise RuntimeError("The efficient SDPA operator is unavailable on this platform.")
        out, lse = _scaled_dot_product_efficient_attention(
            q,
            k,
            v,
            attn_bias=None,
            compute_log_sumexp=True,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    else:
        backend = (torch.nn.attention.SDPBackend.MATH
                   if op_type == "math" else torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
        with torch.nn.attention.sdpa_kernel(backends=[backend]):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
        lse = torch.zeros(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            dtype=q.dtype,
            device=q.device,
        )

    return out.transpose(1, 2), lse.to(q.dtype)


def pytorch_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    block_dq_buffer: torch.Tensor | None = None,
    block_dk_buffer: torch.Tensor | None = None,
    block_dv_buffer: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    bwd_causal: bool | None = None,
    window_size: tuple[int, int] | None = None,
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = True,
    rng_state: torch.Tensor | None = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    del (
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        block_dq_buffer,
        block_dk_buffer,
        block_dv_buffer,
        dropout_p,
        softmax_scale,
        bwd_causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        rng_state,
        args,
        kwargs,
    )
    raise RuntimeError("Backward is not implemented for PyTorch Ring Attention kernels.")


def flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Version-tolerant FlashAttention forward returning output and LSE."""
    del return_softmax
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    out, softmax_lse, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=False,
        return_attn_probs=True,
    )
    return out, softmax_lse


def flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    block_dq_buffer: torch.Tensor,
    block_dk_buffer: torch.Tensor,
    block_dv_buffer: torch.Tensor,
    dropout_p: float,
    softmax_scale: float | None,
    bwd_causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    rng_state: torch.Tensor,
) -> None:
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**-0.5

    # FlashAttention 2.7+ split window_size into left/right positional
    # arguments. FastVideo targets current FlashAttention releases.
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        block_dq_buffer,
        block_dk_buffer,
        block_dv_buffer,
        dropout_p,
        softmax_scale,
        bwd_causal,
        window_size[0],
        window_size[1],
        softcap,
        alibi_slopes,
        deterministic,
        rng_state,
    )


def flash_attn3_func_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del dropout_p, alibi_slopes, return_softmax
    assert HAS_FLASH_ATTN_HOPPER, "FlashAttention Hopper is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**-0.5
    out, softmax_lse, *_ = flash_attn_forward_hopper(
        q=q,
        k=k,
        v=v,
        k_new=None,
        v_new=None,
        qv=None,
        out=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        page_table=None,
        kv_batch_idx=None,
        leftpad_k=None,
        rotary_cos=None,
        rotary_sin=None,
        seqlens_rotary=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_chunk=0,
        softcap=softcap,
        rotary_interleaved=True,
        scheduler_metadata=None,
        num_splits=0,
        pack_gqa=None,
        sm_margin=0,
    )
    return out, softmax_lse


def flash_attn3_func_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    block_dq_buffer: torch.Tensor,
    block_dk_buffer: torch.Tensor,
    block_dv_buffer: torch.Tensor,
    dropout_p: float,
    softmax_scale: float | None,
    bwd_causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    rng_state: torch.Tensor | None,
) -> None:
    del dropout_p, alibi_slopes, rng_state
    assert HAS_FLASH_ATTN_HOPPER, "FlashAttention Hopper is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**-0.5
    flash_attn_func_hopper_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dq=block_dq_buffer,
        dk=block_dk_buffer,
        dv=block_dv_buffer,
        softmax_scale=softmax_scale,
        causal=bwd_causal,
        window_size=window_size,
        softcap=softcap,
        deterministic=deterministic,
        sm_margin=0,
    )


def flash_attn_forward_aiter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del softcap, return_softmax
    assert HAS_AITER, "Aiter is not available"
    return flash_attn_func_aiter(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_lse=True,
    )


def flashinfer_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del dropout_p, alibi_slopes, return_softmax
    assert HAS_FLASHINFER, "FlashInfer is not available"

    kwargs = {
        "sm_scale": softmax_scale,
        "causal": causal,
        "logits_soft_cap": 0.0 if softcap is None else softcap,
        "window_left": window_size[0],
        "return_lse": True,
    }
    if q.ndim == 4:
        if q.shape[0] != 1:
            raise ValueError("FlashInfer Ring Attention only supports batch size 1.")
        out, lse = single_prefill_with_kv_cache(q[0], k[0], v[0], **kwargs)
        out = out.unsqueeze(0)
        lse = lse.transpose(0, 1).unsqueeze(0)
    elif q.ndim == 3:
        out, lse = single_prefill_with_kv_cache(q, k, v, **kwargs)
        lse = lse.transpose(0, 1)
    else:
        raise ValueError(f"Invalid FlashInfer input shape: {tuple(q.shape)}")
    return out, lse / _LOG2_E


def flashinfer_attn_backbward(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise RuntimeError("Backward is not implemented for FlashInfer Ring Attention.")


def npu_fused_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_num: int | None = None,
    input_layout: str = "BSND",
    scale: float | None = None,
    pre_tokens: int = 65535,
    next_tokens: int = 65535,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert HAS_NPU, "torch_npu is not available"
    attention_out, softmax_max, softmax_sum, *_ = torch_npu.npu_fusion_attention_v2(
        q,
        k,
        v,
        head_num=head_num,
        input_layout=input_layout,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
    )
    return attention_out, softmax_max, softmax_sum


def npu_fused_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_attention_out: torch.Tensor,
    head_num: int | None = None,
    input_layout: str = "BSND",
    softmax_max: torch.Tensor | None = None,
    softmax_sum: torch.Tensor | None = None,
    attention_in: torch.Tensor | None = None,
    scale_value: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert HAS_NPU, "torch_npu is not available"
    dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad_v2(
        q,
        k,
        v,
        grad_attention_out,
        head_num,
        input_layout,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=attention_in,
        scale_value=scale_value,
    )
    return dq, dk, dv


__all__ = [
    "flash_attn3_func_backward",
    "flash_attn3_func_forward",
    "flash_attn_backward",
    "flash_attn_forward",
    "flash_attn_forward_aiter",
    "flashinfer_attn_backbward",
    "flashinfer_attn_forward",
    "npu_fused_attn_backward",
    "npu_fused_attn_forward",
    "pytorch_attn_backward",
    "pytorch_attn_forward",
]
