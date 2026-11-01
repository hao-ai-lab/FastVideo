# SPDX-License-Identifier: Apache-2.0
"""cuDNN fused scaled-dot-product attention for dense ``[B, S, H, D]`` tensors.

Drop-in alternative to ``flash_attn_func`` for the no-mask, inference-only dense
self-attention call. It keeps the same precision contract (bf16/fp16 operands,
fp32 softmax/accumulation, bf16/fp16 output) but uses cuDNN's attention engine,
which has a different reduction order than FlashAttention.

The ATen entry point has no shape metadata: it always reads its inputs as
``[B, H, S, D]``. Handing it the caller's contiguous ``[B, S, H, D]`` tensors is
silently misread -- when ``H`` and ``S`` swap into the same output shape the
call even returns a correctly-shaped tensor -- so every call here goes through
``transpose(1, 2)`` to present a BHSD *strided view*, exactly like the
documented PyTorch usage. Never pass contiguous BSHD to the op directly.
"""

from __future__ import annotations

import torch

# cuDNN's fused attention kernels take fp16/bf16 only; the H3 DiT runs bf16.
_CUDNN_SDPA_DTYPES = (torch.float16, torch.bfloat16)
# cuDNN 9.x fused SDPA supports head sizes up to 128.
_CUDNN_SDPA_MAX_HEAD_DIM = 128

# Older torch builds may not register the op.
_ATEN_CUDNN_SDPA = getattr(torch.ops.aten, "_scaled_dot_product_cudnn_attention", None)
if _ATEN_CUDNN_SDPA is not None:
    _ATEN_CUDNN_SDPA = _ATEN_CUDNN_SDPA.default


def cudnn_sdpa_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    """Whether the dense cuDNN route can take this Q/K/V triple unchanged.

    Grad-enabled calls stay on the training path, and calls inside a compiled
    region fall through to the backend's traceable FlashAttention route so the
    opt-in never changes what ``torch.compile`` can capture.
    """
    if _ATEN_CUDNN_SDPA is None:
        return False
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.dtype not in _CUDNN_SDPA_DTYPES or key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        return False
    if query.shape[-1] > _CUDNN_SDPA_MAX_HEAD_DIM:
        return False
    # The BHSD view below only describes the intended (B, S, H, D) layout when
    # the caller's heads are head-dim contiguous; require that explicitly.
    return query.is_contiguous() and key.is_contiguous() and value.is_contiguous()


def cudnn_sdpa_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    causal: bool = False,
) -> torch.Tensor:
    """Dense scaled-dot-product attention on contiguous ``[B, S, H, D]`` inputs."""
    if _ATEN_CUDNN_SDPA is None:
        raise RuntimeError("torch build does not provide aten::_scaled_dot_product_cudnn_attention")
    output = _ATEN_CUDNN_SDPA(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        None,
        False,
        0.0,
        causal,
        False,
        scale=scale,
    )[0]
    return output.transpose(1, 2).contiguous()
