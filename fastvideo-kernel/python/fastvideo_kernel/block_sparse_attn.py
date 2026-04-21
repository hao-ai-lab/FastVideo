"""Autograd-enabled block-sparse attention. Index-native ops with a bool-mask compat shim."""

from __future__ import annotations

import os
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Backend selection helpers
# ---------------------------------------------------------------------------


def _get_sm90_ops():
    try:
        from fastvideo_kernel._C import fastvideo_kernel_ops  # type: ignore
    except Exception:
        return None, None
    return (
        getattr(fastvideo_kernel_ops, "block_sparse_fwd", None),
        getattr(fastvideo_kernel_ops, "block_sparse_bwd", None),
    )


def _is_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return major == 9 and minor == 0


def _force_triton() -> bool:
    return os.environ.get("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", "0") == "1"


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _map_to_index(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compact a bool block_map to (q2k_idx, q2k_num). Legacy path only."""
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    if block_map.dim() != 4:
        raise ValueError(
            f"block_map must be [B,H,Q,KV] (or [H,Q,KV]), "
            f"got shape={tuple(block_map.shape)}"
        )
    if block_map.dtype != torch.bool:
        block_map = block_map.to(torch.bool)
    if not block_map.is_cuda:
        raise RuntimeError(
            "block_map must be a CUDA tensor (Triton map_to_index required)."
        )

    try:
        from fastvideo_kernel.triton_kernels.index import map_to_index as triton_map_to_index
    except Exception as e:  # pragma: no cover - environment issue
        raise ImportError(
            "Triton map_to_index is required but not available. "
            "Ensure Triton is installed and "
            "fastvideo_kernel.triton_kernels.index is importable."
        ) from e
    return triton_map_to_index(block_map)


def _invert_indices_for_backward(
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    num_kv_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from fastvideo_kernel.triton_kernels.index import invert_indices
    return invert_indices(q2k_idx, q2k_num, num_kv_blocks=num_kv_blocks)


# ---------------------------------------------------------------------------
# Triton backend custom ops (index-native)
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_triton",
    mutates_args=(),
    device_types="cuda",
)
def block_sparse_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import (
        triton_block_sparse_attn_forward,
    )

    o, M = triton_block_sparse_attn_forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        q2k_idx,
        q2k_num,
        variable_block_sizes,
    )
    return o, M


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_triton")
def _block_sparse_attn_triton_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty_like(q)
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]),
        device=q.device,
        dtype=torch.float32,
    )
    return o, M


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_backward_triton",
    mutates_args=(),
    device_types="cuda",
)
def block_sparse_attn_backward_triton(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    M: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import (
        triton_block_sparse_attn_backward,
    )

    num_kv_blocks = int(variable_block_sizes.numel())
    k2q_idx, k2q_num = _invert_indices_for_backward(
        q2k_idx, q2k_num, num_kv_blocks
    )
    dq, dk, dv = triton_block_sparse_attn_backward(
        grad_output.contiguous(),
        q,
        k,
        v,
        o,
        M,
        q2k_idx,
        q2k_num,
        k2q_idx,
        k2q_num,
        variable_block_sizes,
    )
    return dq, dk, dv


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_backward_triton")
def _block_sparse_attn_backward_triton_fake(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    M: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    return dq, dk, dv


def _setup_context_triton(ctx, inputs, output):
    q, k, v, q2k_idx, q2k_num, variable_block_sizes = inputs
    o, M = output
    ctx.save_for_backward(q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes)


def _backward_triton(ctx, grad_o, grad_M):
    q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes = ctx.saved_tensors
    dq, dk, dv = block_sparse_attn_backward_triton(
        grad_o, q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes
    )
    return dq, dk, dv, None, None, None


block_sparse_attn_triton.register_autograd(
    _backward_triton, setup_context=_setup_context_triton
)


# ---------------------------------------------------------------------------
# SM90 backend custom ops (index-native)
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_sm90",
    mutates_args=(),
    device_types="cuda",
)
def block_sparse_attn_sm90(
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_sparse_fwd, _ = _get_sm90_ops()
    if block_sparse_fwd is None:
        raise ImportError("fastvideo_kernel_ops.block_sparse_fwd is not available")

    o_padded, lse_padded = block_sparse_fwd(
        q_padded.contiguous(),
        k_padded.contiguous(),
        v_padded.contiguous(),
        q2k_idx,
        q2k_num,
        variable_block_sizes.int(),
    )
    return o_padded, lse_padded


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_sm90")
def _block_sparse_attn_sm90_fake(
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty_like(q_padded)
    lse = torch.empty(
        (q_padded.shape[0], q_padded.shape[1], q_padded.shape[2], 1),
        device=q_padded.device,
        dtype=torch.float32,
    )
    return o, lse


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_backward_sm90",
    mutates_args=(),
    device_types="cuda",
)
def block_sparse_attn_backward_sm90(
    grad_output_padded: torch.Tensor,
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    o_padded: torch.Tensor,
    lse_padded: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, block_sparse_bwd = _get_sm90_ops()
    if block_sparse_bwd is None:
        raise ImportError("fastvideo_kernel_ops.block_sparse_bwd is not available")

    num_kv_blocks = int(variable_block_sizes.numel())
    k2q_idx, k2q_num = _invert_indices_for_backward(
        q2k_idx, q2k_num, num_kv_blocks
    )

    dq, dk, dv = block_sparse_bwd(
        q_padded,
        k_padded,
        v_padded,
        o_padded,
        lse_padded,
        grad_output_padded.contiguous(),
        k2q_idx,
        k2q_num,
        variable_block_sizes.int(),
    )
    # C++ kernel returns fp32 grads; cast back to the input dtype.
    out_dtype = grad_output_padded.dtype
    return dq.to(out_dtype), dk.to(out_dtype), dv.to(out_dtype)


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_backward_sm90")
def _block_sparse_attn_backward_sm90_fake(
    grad_output_padded: torch.Tensor,
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    o_padded: torch.Tensor,
    lse_padded: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q_padded)
    dk = torch.empty_like(k_padded)
    dv = torch.empty_like(v_padded)
    return dq, dk, dv


def _setup_context_sm90(ctx, inputs, output):
    q, k, v, q2k_idx, q2k_num, variable_block_sizes = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, q2k_idx, q2k_num, variable_block_sizes)


def _backward_sm90(ctx, grad_o, grad_lse):
    q, k, v, o, lse, q2k_idx, q2k_num, variable_block_sizes = ctx.saved_tensors
    dq, dk, dv = block_sparse_attn_backward_sm90(
        grad_o, q, k, v, o, lse, q2k_idx, q2k_num, variable_block_sizes
    )
    return dq, dk, dv, None, None, None


block_sparse_attn_sm90.register_autograd(
    _backward_sm90, setup_context=_setup_context_sm90
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def block_sparse_attn_from_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-sparse attention with autograd, taking compact per-row KV indices."""
    block_sparse_fwd, block_sparse_bwd = _get_sm90_ops()
    use_sm90 = (
        (not _force_triton())
        and _is_sm90()
        and block_sparse_fwd is not None
        and block_sparse_bwd is not None
    )
    if use_sm90:
        return block_sparse_attn_sm90(q, k, v, q2k_idx, q2k_num, variable_block_sizes)
    # Triton path: supports q_seq_len != kv_seq_len as long as both are padded
    # to a multiple of the block size (64 tokens).
    return block_sparse_attn_triton(q, k, v, q2k_idx, q2k_num, variable_block_sizes)


def block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bool-mask compat wrapper; prefer block_sparse_attn_from_indices."""
    q2k_idx, q2k_num = _map_to_index(block_map)
    return block_sparse_attn_from_indices(
        q, k, v, q2k_idx, q2k_num, variable_block_sizes
    )
