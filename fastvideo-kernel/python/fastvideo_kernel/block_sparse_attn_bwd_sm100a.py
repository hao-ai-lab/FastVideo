# SPDX-License-Identifier: Apache-2.0
"""sm_100a (Blackwell) CUDA block-sparse VSA backward.

Companion of ``block_sparse_attn_sm100a`` (the forward): consumes the forward's ``lse`` in the
Triton "M format" (``max(qk * sm_scale * log2e) + log2(l)``, ``[B, H, S]`` fp32) unchanged and
FastVideo's k2q index metadata, returns ``(dq, dk, dv)`` in bf16 with the inputs' layout and the
Triton backward's scaling (dq and dk carry sm_scale, dv does not). 64-token blocks only; any
other configuration falls back to Triton via ``is_supported``.
"""

from typing import Tuple

import torch

try:
    # The pybind symbols live on fastvideo_kernel_ops, NOT on the _C package that contains it
    # (its __init__ is empty, so hasattr on the package fails with the kernel built and present).
    from fastvideo_kernel._C import fastvideo_kernel_ops as _C
    _BWD = getattr(_C, "block_sparse_sm100a_bwd", None)
    _HAS_VSA_BWD_SM100A = _BWD is not None
except ImportError:  # pragma: no cover - extension not built
    _C = None
    _BWD = None
    _HAS_VSA_BWD_SM100A = False

_SM100 = (10, 0)
HEAD_DIM = 128
BLOCK = 64
# Must match the -DVSA_BHSD the extension was compiled with (FastVideo builds with true).
BHSD = True


def set_extension(module) -> None:
    """Use an already-loaded extension module exposing ``block_sparse_sm100a_bwd``.

    The in-repo tests build the binding standalone (tests/jit_ext.py) and inject it here, so
    the backend can be exercised without a fastvideo_kernel install.
    """
    global _C, _BWD, _HAS_VSA_BWD_SM100A
    _C = module
    _BWD = getattr(module, "block_sparse_sm100a_bwd", None)
    _HAS_VSA_BWD_SM100A = _BWD is not None


def _seqlen(q: torch.Tensor) -> int:
    return q.shape[2] if BHSD else q.shape[1]


def is_supported(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> bool:
    """True iff this build can run these tensors; otherwise the caller uses Triton.

    Static facts only (shapes, dtypes, arch, layout), never tensor contents, so it is cheap
    enough for a per-layer dispatch path. The kernel is fixed at 64-token blocks with
    head_dim 128 and needs seqlen == 64 * num_blocks with an even num_blocks (its preprocess
    works in 128-token blocks). Per-row k2q counts may be anything in [0, num_q_blocks],
    including 0: unselected kv blocks get exactly-zero dk/dv rows.
    """
    if not _HAS_VSA_BWD_SM100A or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) != _SM100:
        return False
    if q.dtype != torch.bfloat16 or q.dim() != 4 or q.shape[-1] != HEAD_DIM:
        return False
    if not q.is_contiguous():
        return False
    # Metadata must be integer-typed so the wrapper's int32 conversion is value-preserving.
    if not variable_block_sizes.is_cuda or variable_block_sizes.dtype not in (torch.int32,
                                                                              torch.int64):
        return False
    num_blocks = variable_block_sizes.numel()
    if num_blocks == 0 or num_blocks % 2 != 0:
        return False
    if _seqlen(q) != BLOCK * num_blocks:
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

    ``k2q_idx`` is ``[B, H, num_kv_blocks, max_q_blocks]`` (or the flat 2-D view) of LOCAL q64
    block ids, ``k2q_num`` ``[B, H, num_kv_blocks]``; entries past a row's count are never read.
    """
    sm_scale = 1.0 / (q.shape[-1]**0.5)
    idx = k2q_idx.to(torch.int32).contiguous()
    num = k2q_num.to(torch.int32).contiguous()
    vbs = variable_block_sizes.to(torch.int32).contiguous()
    res = _BWD(grad_o.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(),
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
