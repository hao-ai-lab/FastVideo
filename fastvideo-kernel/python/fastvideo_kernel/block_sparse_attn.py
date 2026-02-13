from __future__ import annotations

import os

import torch


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
    # Force Triton even on SM90 and even if the compiled extension is available.
    # Useful for CI / debugging / parity testing.
    return os.environ.get("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", "0") == "1"


def _map_to_index(block_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Preferred map->index conversion used by the wrapper.

    This wrapper **requires** the Triton implementation.
    If Triton (or the Triton map_to_index module) is not available, it raises.
    """
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    if block_map.dim() != 4:
        raise ValueError(f"block_map must be [B,H,Q,KV] (or [H,Q,KV]), got shape={tuple(block_map.shape)}")
    if block_map.dtype != torch.bool:
        block_map = block_map.to(torch.bool)

    if not block_map.is_cuda:
        raise RuntimeError("block_map must be a CUDA tensor (Triton map_to_index required).")

    try:
        from fastvideo_kernel.triton_kernels.index import map_to_index as triton_map_to_index  # local import
    except Exception as e:
        raise ImportError(
            "Triton map_to_index is required but not available. "
            "Ensure Triton is installed and fastvideo_kernel.triton_kernels.index is importable."
        ) from e
    return triton_map_to_index(block_map)


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_triton",
    mutates_args=(),
    device_types="cuda",
)
def block_sparse_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_map = block_map.to(torch.bool)
    q2k_idx, q2k_num = _map_to_index(block_map)

    from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import (  # local import
        triton_block_sparse_attn_forward,
    )

    o, M = triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, variable_block_sizes)
    return o, M



@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_triton")
def _block_sparse_attn_triton_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
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
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    block_map = block_map.to(torch.bool)
    q2k_idx, q2k_num = _map_to_index(block_map)
    k2q_idx, k2q_num = _map_to_index(block_map.transpose(-1, -2).contiguous())

    from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import (  # local import
        triton_block_sparse_attn_backward,
    )

    dq, dk, dv = triton_block_sparse_attn_backward(
        grad_output, q, k, v, o, M, q2k_idx, q2k_num, k2q_idx, k2q_num, variable_block_sizes
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
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    return dq, dk, dv


def _backward_triton(ctx, grad_o, grad_M):
    q, k, v, o, M, block_map, variable_block_sizes = ctx.saved_tensors
    dq, dk, dv = block_sparse_attn_backward_triton(grad_o, q, k, v, o, M, block_map, variable_block_sizes)
    return dq, dk, dv, None, None


def _setup_context_triton(ctx, inputs, output):
    q, k, v, block_map, variable_block_sizes = inputs
    o, M = output
    ctx.save_for_backward(q, k, v, o, M, block_map, variable_block_sizes)


block_sparse_attn_triton.register_autograd(_backward_triton, setup_context=_setup_context_triton)


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_sm90",
    mutates_args=(),
    device_types="cuda",
)
def block_sparse_attn_sm90(
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_sparse_fwd, _ = _get_sm90_ops()
    if block_sparse_fwd is None:
        raise ImportError("fastvideo_kernel_ops.block_sparse_fwd is not available")

    q_padded = q_padded.contiguous()
    k_padded = k_padded.contiguous()
    v_padded = v_padded.contiguous()
    block_map = block_map.to(torch.bool)
    q2k_idx, q2k_num = _map_to_index(block_map)

    o_padded, lse_padded = block_sparse_fwd(
        q_padded, k_padded, v_padded, q2k_idx, q2k_num, variable_block_sizes.int()
    )
    return o_padded, lse_padded


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_sm90")
def _block_sparse_attn_sm90_fake(
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty_like(q_padded)
    lse = torch.empty((q_padded.shape[0], q_padded.shape[1], q_padded.shape[2], 1), device=q_padded.device, dtype=torch.float32)
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
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, block_sparse_bwd = _get_sm90_ops()
    if block_sparse_bwd is None:
        raise ImportError("fastvideo_kernel_ops.block_sparse_bwd is not available")

    grad_output_padded = grad_output_padded.contiguous()
    block_map = block_map.to(torch.bool)
    k2q_idx, k2q_num = _map_to_index(block_map.transpose(-1, -2).contiguous())

    dq, dk, dv = block_sparse_bwd(
        q_padded,
        k_padded,
        v_padded,
        o_padded,
        lse_padded,
        grad_output_padded,
        k2q_idx,
        k2q_num,
        variable_block_sizes.int(),
    )
    # C++ kernel returns fp32 grads; cast back to match PyTorch convention if needed
    return dq.to(grad_output_padded.dtype), dk.to(grad_output_padded.dtype), dv.to(grad_output_padded.dtype)


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_backward_sm90")
def _block_sparse_attn_backward_sm90_fake(
    grad_output_padded: torch.Tensor,
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    o_padded: torch.Tensor,
    lse_padded: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q_padded)
    dk = torch.empty_like(k_padded)
    dv = torch.empty_like(v_padded)
    return dq, dk, dv


def _backward_sm90(ctx, grad_o, grad_lse):
    q, k, v, o, lse, block_map, variable_block_sizes = ctx.saved_tensors
    dq, dk, dv = block_sparse_attn_backward_sm90(
        grad_o, q, k, v, o, lse, block_map, variable_block_sizes
    )
    return dq, dk, dv, None, None


def _setup_context_sm90(ctx, inputs, output):
    q, k, v, block_map, variable_block_sizes = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, block_map, variable_block_sizes)


block_sparse_attn_sm90.register_autograd(_backward_sm90, setup_context=_setup_context_sm90)


def block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unified block-sparse attention op with autograd support.
    - On SM90 with compiled extension present: uses fastvideo_kernel_ops.block_sparse_fwd/bwd.
    - Otherwise: uses Triton implementation (requires q/k/v to have same padded length today).
    """
    block_sparse_fwd, block_sparse_bwd = _get_sm90_ops()
    if (not _force_triton()) and _is_sm90() and (block_sparse_fwd is not None) and (block_sparse_bwd is not None):
        return block_sparse_attn_sm90(q, k, v, block_map, variable_block_sizes)
    # Triton path: generally assumes q/k/v share the same padded length
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise RuntimeError("Triton fallback requires q/k/v to have the same padded length.")
    return block_sparse_attn_triton(q, k, v, block_map, variable_block_sizes)


