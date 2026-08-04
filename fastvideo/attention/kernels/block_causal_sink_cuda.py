# SPDX-License-Identifier: Apache-2.0
"""CUDA (sm_100a) forward for the block-causal + sink + sliding-window training attention.

Goes in FastVideo as ``fastvideo/attention/kernels/block_causal_sink_cuda.py``.

This replaces ONLY the forward. It returns ``out`` and ``lse`` in exactly the form
``_fwd_kernel`` produces them, so ``_BlockCausalSinkAttention.backward`` and its Triton
kernels are reused untouched -- see INTEGRATION.md for the ~6-line patch to
``block_causal_sink.py``.

Scope: ``kind="blockwise"``, Blackwell (sm_100a), bf16, ``head_dim == 128``, uniform
sequence length, ``num_frames % num_frame_per_block == 0``, and a sink reaching at most one
128-token tile past a block end. Everything else must fall back to Triton -- outside that
regime the reference is self-inconsistent, so returning an answer at all would be wrong.
``is_supported()`` is the predicate; callers should consult it rather than assume.
"""
from __future__ import annotations

import torch

try:
    import fastvideo_kernel._C as _C
    _HAS_CUDA_BCS = hasattr(_C, "block_causal_sink_sm100a_fwd")
except ImportError:  # pragma: no cover - extension not built
    _C = None
    _HAS_CUDA_BCS = False

_SM100 = (10, 0)  # compiled for Blackwell only
HEAD_DIM = 128
K_TILE = 128


def is_supported(plan, q: torch.Tensor) -> bool:
    """True iff this backend can run `plan` on `q`; otherwise the caller uses Triton."""
    if not _HAS_CUDA_BCS or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) != _SM100:
        return False
    if plan.kind != "blockwise":
        return False  # teacher_forcing is a separate kernel
    if q.dtype != torch.bfloat16 or q.shape[-1] != HEAD_DIM:
        return False
    if q.stride(-1) != 1:
        return False  # head_dim contiguous; outer strides are free
    if plan.local_attn_size is None or plan.local_attn_size < 0:
        return False
    if plan.num_frames % plan.num_frame_per_block != 0:
        return False  # partial last block is out of spec
    if (plan.sink_size - plan.num_frame_per_block) * plan.frame_seqlen > K_TILE:
        return False  # large-sink regime is out of spec
    return plan.num_frame_per_block * plan.frame_seqlen > 0


def block_causal_sink_forward_cuda(q, k, v, q_sink, plan):
    """Forward pass. q/k/v/q_sink: ``[B, H, L, D]`` bf16. Returns ``(out, lse)``.

    Tensors are consumed with whatever strides they arrive with -- a permuted view is fine
    and is NOT copied; only ``head_dim`` has to be the contiguous axis. ``lse`` is
    ``[B*H, L]`` float32, identical in meaning to the Triton forward's.
    """
    out, lse = _C.block_causal_sink_sm100a_fwd(
        q,
        k,
        v,
        q_sink if q_sink is not None and q_sink is not q else None,
        plan.num_frame_per_block * plan.frame_seqlen,  # tokens_per_block
        plan.sink_size * plan.frame_seqlen,  # sink_tokens
        plan.local_attn_size * plan.frame_seqlen,  # rolling_window_tokens
        float(plan.sm_scale),
        True,  # need_lse (backward consumes it)
    )
    return out, lse
