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
    """True iff Triton is explicitly forced via env var.

    Honors both the new ``FASTVIDEO_VSA_TRITON`` and the legacy
    ``FASTVIDEO_KERNEL_VSA_FORCE_TRITON`` for backward compatibility.
    """
    return (os.environ.get("FASTVIDEO_VSA_TRITON", "0") == "1"
            or os.environ.get("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", "0") == "1")


def _force_tk() -> bool:
    """True iff the sm_90 (ThunderKittens) backend is explicitly requested.

    Only effective on sm_90 with the compiled extension present; otherwise
    falls back to the default selection.
    """
    return os.environ.get("FASTVIDEO_VSA_TK", "0") == "1"


def _force_sm100a() -> bool:
    """True iff the sm_100a (Blackwell) forward is explicitly opted into.

    Opt-in only (same env the H3 backend honors): the sm_100a extension is
    forward-only, so this routing pairs it with the Triton backward -- its lse
    is already in Triton's M format. Honored only when
    ``block_sparse_attn_sm100a.is_supported`` passes. Unsupported 64-token
    metadata falls through to the default selection; unsupported 128-token
    metadata raises because Triton has no compatible fallback.
    ``FASTVIDEO_VSA_TRITON`` still wins.
    """
    return os.environ.get("FASTVIDEO_VSA_SM100A", "0") == "1"


def _sm100a_is_supported(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> bool:
    try:
        from fastvideo_kernel import block_sparse_attn_sm100a as vsa_sm100a
    except Exception:
        return False
    return vsa_sm100a.is_supported(q, variable_block_sizes)


def _infer_block_size(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> int:
    num_blocks = variable_block_sizes.numel()
    seq_len = q.shape[2]
    if num_blocks == 0 or seq_len % num_blocks != 0:
        return 0
    return seq_len // num_blocks


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _map_to_index(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compact a bool block_map to (q2k_idx, q2k_num). Legacy path only."""
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    if block_map.dim() != 4:
        raise ValueError(f"block_map must be [B,H,Q,KV] (or [H,Q,KV]), "
                         f"got shape={tuple(block_map.shape)}")
    if block_map.dtype != torch.bool:
        block_map = block_map.to(torch.bool)
    if not block_map.is_cuda:
        raise RuntimeError("block_map must be a CUDA tensor (Triton map_to_index required).")

    try:
        from fastvideo_kernel.triton_kernels.index import map_to_index as triton_map_to_index
    except Exception as e:  # pragma: no cover - environment issue
        raise ImportError("Triton map_to_index is required but not available. "
                          "Ensure Triton is installed and "
                          "fastvideo_kernel.triton_kernels.index is importable.") from e
    return triton_map_to_index(block_map)


def _invert_indices_for_backward(
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    num_kv_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from fastvideo_kernel.triton_kernels.index import invert_indices
    return invert_indices(q2k_idx, q2k_num, num_kv_blocks=num_kv_blocks)


def _as_int32_contig(t: torch.Tensor, name: str) -> torch.Tensor:
    """Return `t` as a contiguous int32 tensor, raising a clear error on CPU input."""
    if not t.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor, got device={t.device}")
    if t.dtype != torch.int32:
        t = t.to(torch.int32)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


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
        triton_block_sparse_attn_forward, )

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
        triton_block_sparse_attn_backward, )

    num_kv_blocks = int(variable_block_sizes.numel())
    k2q_idx, k2q_num = _invert_indices_for_backward(q2k_idx, q2k_num, num_kv_blocks)
    # q/k/v are saved from the user-facing inputs and may be non-contiguous;
    # o/M are kernel outputs so are already contiguous.
    dq, dk, dv = triton_block_sparse_attn_backward(
        grad_output.contiguous(),
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
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
    dq, dk, dv = block_sparse_attn_backward_triton(grad_o, q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes)
    return dq, dk, dv, None, None, None


block_sparse_attn_triton.register_autograd(_backward_triton, setup_context=_setup_context_triton)

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
        variable_block_sizes,
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
    k2q_idx, k2q_num = _invert_indices_for_backward(q2k_idx, q2k_num, num_kv_blocks)

    # q/k/v are saved from user-facing inputs; o/lse are kernel outputs.
    dq, dk, dv = block_sparse_bwd(
        q_padded.contiguous(),
        k_padded.contiguous(),
        v_padded.contiguous(),
        o_padded,
        lse_padded,
        grad_output_padded.contiguous(),
        k2q_idx,
        k2q_num,
        variable_block_sizes,
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
    dq, dk, dv = block_sparse_attn_backward_sm90(grad_o, q, k, v, o, lse, q2k_idx, q2k_num, variable_block_sizes)
    return dq, dk, dv, None, None, None


block_sparse_attn_sm90.register_autograd(_backward_sm90, setup_context=_setup_context_sm90)

# ---------------------------------------------------------------------------
# SM100A backend custom op (index-native)
#
# Forward runs the sm_100a CUDA extension; backward reuses the Triton kernels.
# The sm_100a forward emits lse in exactly Triton's M format (max*log2e +
# log2(l)), so the pairing needs no conversion. The Triton backward is
# hardcoded to 64-token blocks, hence the block-size assert below.
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_sm100a",
    mutates_args=(),
    device_types="cuda",
)
def block_sparse_attn_sm100a_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from fastvideo_kernel.block_sparse_attn_sm100a import block_sparse_attn_sm100a

    o, M = block_sparse_attn_sm100a(q, k, v, q2k_idx, q2k_num, variable_block_sizes,
                                    need_lse=True)
    return o, M


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_sm100a")
def _block_sparse_attn_sm100a_fake(
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


def _setup_context_sm100a(ctx, inputs, output):
    q, k, v, q2k_idx, q2k_num, variable_block_sizes = inputs
    o, M = output
    ctx.save_for_backward(q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes)


def _backward_sm100a(ctx, grad_o, grad_M):
    q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes = ctx.saved_tensors
    block = q.shape[2] // variable_block_sizes.numel()
    if block != 64:
        raise RuntimeError(
            "block_sparse_attn_sm100a backward pairs the sm_100a forward with the "
            f"Triton backward, which is hardcoded to 64-token blocks; got {block}. "
            "Run 128-token-block metadata without grad, or use the Triton forward.")
    dq, dk, dv = block_sparse_attn_backward_triton(grad_o, q, k, v, o, M, q2k_idx,
                                                   q2k_num, variable_block_sizes)
    return dq, dk, dv, None, None, None


block_sparse_attn_sm100a_op.register_autograd(_backward_sm100a,
                                              setup_context=_setup_context_sm100a)

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
    # Normalize index tensors once at the public boundary so the custom ops
    # and their fakes can assume int32/contiguous. No-op on well-formed input.
    q2k_idx = _as_int32_contig(q2k_idx, "q2k_idx")
    q2k_num = _as_int32_contig(q2k_num, "q2k_num")
    variable_block_sizes = _as_int32_contig(variable_block_sizes, "variable_block_sizes")

    block_sparse_fwd, block_sparse_bwd = _get_sm90_ops()
    sm90_available = (_is_sm90() and block_sparse_fwd is not None and block_sparse_bwd is not None)

    # Backend resolution:
    # - FASTVIDEO_VSA_TRITON forces Triton everywhere.
    # - FASTVIDEO_VSA_SM100A opts into the sm_100a forward (Triton backward).
    #   Unsupported 64-token metadata falls through; unsupported 128-token
    #   metadata raises because Triton cannot consume it.
    # - FASTVIDEO_VSA_TK requests sm_90 TK; honored only when it's actually
    #   available (else falls through to the default below).
    # - Otherwise: TK on sm_90 if available, else Triton.
    if _force_triton():
        use_sm90 = False
    elif _force_sm100a():
        if _sm100a_is_supported(q, variable_block_sizes):
            return block_sparse_attn_sm100a_op(q, k, v, q2k_idx, q2k_num,
                                               variable_block_sizes)
        if _infer_block_size(q, variable_block_sizes) == 128:
            raise NotImplementedError(
                "128-token block-sparse attention requires the sm_100a forward; "
                "the Triton fallback only supports 64-token blocks, and the "
                "sm_100a route is unavailable for this input.")
        use_sm90 = sm90_available
    elif _force_tk():
        use_sm90 = sm90_available
    else:
        use_sm90 = sm90_available

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
    return block_sparse_attn_from_indices(q, k, v, q2k_idx, q2k_num, variable_block_sizes)
