# SPDX-License-Identifier: Apache-2.0
"""sm_100a/sm_103a (data-center Blackwell) CUDA block-sparse VSA backward.

Companion of ``block_sparse_attn_sm100a`` (the forward): consumes the forward's ``lse`` in the
Triton "M format" (``max(qk * sm_scale * log2e) + log2(l)``, ``[B, H, S]`` fp32) unchanged and
FastVideo's k2q index metadata, returns ``(dq, dk, dv)`` in bf16 with the inputs' layout and the
Triton backward's scaling (dq and dk carry sm_scale, dv does not). Two kernels, picked by the
metadata's block size: 64-token blocks (``block_sparse_sm100a_bwd``) and 128-token blocks
(``block_sparse_sm100a_blk128_bwd``); any other configuration falls back to Triton via
``is_supported``.
"""

from typing import Tuple

import torch

try:
    # The pybind symbols live on fastvideo_kernel_ops, NOT on the _C package that contains it
    # (its __init__ is empty, so hasattr on the package fails with the kernel built and present).
    from fastvideo_kernel._C import fastvideo_kernel_ops as _C
    _BWD_BY_BLOCK = {
        64: getattr(_C, "block_sparse_sm100a_bwd", None),
        128: getattr(_C, "block_sparse_sm100a_blk128_bwd", None),
    }
    _HAS_VSA_BWD_SM100A = any(_BWD_BY_BLOCK.values())
except ImportError:  # pragma: no cover - extension not built
    _C = None
    _BWD_BY_BLOCK = {}
    _HAS_VSA_BWD_SM100A = False

_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}
HEAD_DIM = 128
BLOCKS = (64, 128)
# Must match the -DVSA_BHSD the extension was compiled with (FastVideo builds with true).
BHSD = True


def set_extension(module) -> None:
    """Use an already-loaded extension module exposing ``block_sparse_sm100a_bwd`` and / or
    ``block_sparse_sm100a_blk128_bwd``.

    A standalone build of the binding .cu files (for example through
    ``torch.utils.cpp_extension.load`` with a ten-line pybind wrapper) can be injected here, so
    the backend can be exercised without rebuilding the fastvideo_kernel wheel.
    """
    global _C, _BWD_BY_BLOCK, _HAS_VSA_BWD_SM100A
    _C = module
    _BWD_BY_BLOCK = {
        64: getattr(module, "block_sparse_sm100a_bwd", None),
        128: getattr(module, "block_sparse_sm100a_blk128_bwd", None),
    }
    _HAS_VSA_BWD_SM100A = any(_BWD_BY_BLOCK.values())


def _seqlen(q: torch.Tensor) -> int:
    return q.shape[2] if BHSD else q.shape[1]


def _block_size(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> int:
    """The block size the metadata implies (seqlen / num_blocks), 0 when it is not integral."""
    num_blocks = variable_block_sizes.numel()
    seqlen = _seqlen(q)
    return 0 if num_blocks == 0 or seqlen % num_blocks else seqlen // num_blocks


def is_supported(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> bool:
    """True iff this build can run these tensors; otherwise the caller uses Triton.

    Static facts only (shapes, dtypes, arch, layout), never tensor contents, so it is cheap
    enough for a per-layer dispatch path. Both kernels take head_dim 128 and seqlen == block *
    num_blocks; the 64-token kernel also needs an even num_blocks (its preprocess works in
    128-token blocks); batch * heads is the launch grid's y extent, so it must be <= 65535.
    Per-row k2q counts may be anything in [0, num_q_blocks], including 0: unselected kv blocks
    get exactly-zero dk/dv rows.
    """
    if not _HAS_VSA_BWD_SM100A or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) not in _SUPPORTED_COMPUTE_CAPABILITIES:
        return False
    if q.dtype != torch.bfloat16 or q.dim() != 4 or q.shape[-1] != HEAD_DIM:
        return False
    if not q.is_contiguous():
        return False
    if q.shape[0] * (q.shape[1] if BHSD else q.shape[2]) > 65535:
        return False
    # Metadata must be integer-typed so the wrapper's int32 conversion is value-preserving.
    if not variable_block_sizes.is_cuda or variable_block_sizes.dtype not in (torch.int32,
                                                                              torch.int64):
        return False
    block = _block_size(q, variable_block_sizes)
    if _BWD_BY_BLOCK.get(block) is None:
        return False
    if block == 64 and variable_block_sizes.numel() % 2 != 0:
        return False
    return True


def block_sparse_attn_backward_sm100a_from_k2q(
    grad_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    k2q_idx: torch.Tensor,
    k2q_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward from k2q metadata already in hand (``invert_indices`` layout).

    ``k2q_idx`` is ``[B, H, num_kv_blocks, max_q_blocks]`` (or the flat 2-D view) of LOCAL q
    block ids (64- or 128-token blocks, as the metadata's block size), ``k2q_num``
    ``[B, H, num_kv_blocks]``; entries past a row's count are never read.
    """
    block = _block_size(q, variable_block_sizes)
    backward = _BWD_BY_BLOCK.get(block)
    if backward is None:
        raise RuntimeError(f"block_sparse_attn_backward_sm100a: no kernel for {block}-token "
                           f"blocks (built: {sorted(b for b, f in _BWD_BY_BLOCK.items() if f)})")
    sm_scale = 1.0 / (q.shape[-1]**0.5)
    idx = k2q_idx.to(torch.int32).contiguous()
    num = k2q_num.to(torch.int32).contiguous()
    vbs = variable_block_sizes.to(torch.int32).contiguous()
    res = backward(grad_o.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(),
                   o.contiguous(), lse.contiguous(), idx, num, vbs, sm_scale)
    return res[0], res[1], res[2]


def block_sparse_attn_backward_sm100a(
    grad_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass from the forward's q2k metadata. Returns ``(dq, dk, dv)``.

    Mirrors ``block_sparse_attn_backward_triton``: the k2q inversion is recomputed here with
    FastVideo's Triton ``invert_indices`` rather than saved by the forward.
    """
    from fastvideo_kernel.triton_kernels.index import invert_indices

    num_kv_blocks = variable_block_sizes.numel()
    batch = q.shape[0]
    heads = q.shape[1] if BHSD else q.shape[2]
    idx = q2k_idx.to(torch.int32).contiguous()
    num = q2k_num.to(torch.int32).contiguous()
    if idx.dim() != 4:
        idx = idx.view(batch, heads, -1, idx.shape[-1])
    if num.dim() != 3:
        num = num.view(batch, heads, -1)
    k2q_idx, k2q_num = invert_indices(idx, num, num_kv_blocks)
    return block_sparse_attn_backward_sm100a_from_k2q(grad_o, q, k, v, o, lse, k2q_idx, k2q_num,
                                                      variable_block_sizes)
