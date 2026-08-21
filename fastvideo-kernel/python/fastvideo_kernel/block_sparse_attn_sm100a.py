# SPDX-License-Identifier: Apache-2.0
"""sm_100a (Blackwell) CUDA block-sparse VSA forward.

A third backend behind the same VSA op as the Triton and CuTe-DSL paths. Forward only: it
returns ``(out, lse)`` with ``lse`` in exactly the form ``triton_block_sparse_attn_forward``
writes -- ``max(qk * qk_scale) + log2(l)``, ``[B, H, S]`` fp32 -- so
``block_sparse_attn_backward_triton`` runs against it unchanged.

The extension carries TWO instantiations of the kernel, for 64- and 128-token sparse blocks
(tile volumes 64 and 128 in ``build_vsa_metadata``); the block size is inferred from the
tensors and picks the op. Anything else falls back to Triton via ``is_supported``.
"""

from typing import Tuple

import torch

try:
    # The pybind symbols live on fastvideo_kernel_ops, NOT on the _C package that contains it.
    # `import fastvideo_kernel._C as _C` resolves to the namespace package, whose __init__ is
    # empty, so hasattr() fails on a wheel install and the caller silently falls back with the
    # kernel built and present.
    from fastvideo_kernel._C import fastvideo_kernel_ops as _C
    _FWD_BY_BLOCK = {
        64: getattr(_C, "block_sparse_sm100a_fwd", None),
        128: getattr(_C, "block_sparse_sm100a_blk128_fwd", None),
    }
    _HAS_VSA_SM100A = any(_FWD_BY_BLOCK.values())
except ImportError:  # pragma: no cover - extension not built
    _C = None
    _FWD_BY_BLOCK = {}
    _HAS_VSA_SM100A = False

_SM100 = (10, 0)
HEAD_DIM = 128
# Must match the -DVSA_BHSD the extension was compiled with (see CMakeLists).
BHSD = True


def _block_size(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> int:
    num_blocks = variable_block_sizes.numel()
    seqlen = q.shape[2] if BHSD else q.shape[1]
    return 0 if num_blocks == 0 or seqlen % num_blocks else seqlen // num_blocks


def is_supported(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> bool:
    """True iff this build can run these tensors; otherwise the caller uses Triton.

    Static facts only -- shapes, dtypes, arch, layout. Deliberately NO reads of tensor
    contents: the previous ``int(variable_block_sizes.min())`` was a GPU->CPU sync on every
    call, and the kernel no longer needs it (see below). This predicate must stay cheap
    enough to sit on a per-layer dispatch path.

    What the kernel accepts (and is tested to handle):
      * q/k/v: contiguous 4-D bf16 CUDA tensors on an sm_100 device, head_dim 128, laid out
        as compiled (BHSD here); seqlen == num_blocks * block with an EVEN num_blocks (a CTA
        owns an adjacent pair of query blocks) and a 64- or 128-token build present.
      * q2k_num: any per-row counts in [0, max_kv], NON-uniform across rows included. Rows
        with count 0 produce exactly-zero output rows (and a finite LSE sentinel) rather
        than attending anywhere -- so no ``.min()`` floor is required of the caller.
      * q2k_idx: rows only need valid entries (in [0, num_blocks)) BELOW that row's count;
        padding past the count (e.g. map_to_index's -1 fill) is never dereferenced. max_kv
        (= q2k_idx.shape[-1]) must be >= 1, which the host launcher re-checks.
      * variable_block_sizes: per-KV-block valid-token counts in [0, block]; keys at or past
        a block's count are masked. Integer metadata is converted to int32/contiguous by
        ``block_sparse_attn_sm100a`` itself, so int64 inputs merely cost a cast.
    """
    if not _HAS_VSA_SM100A or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) != _SM100:
        return False
    if q.dtype != torch.bfloat16 or q.dim() != 4 or q.shape[-1] != HEAD_DIM:
        return False
    if not q.is_contiguous():
        return False
    if _FWD_BY_BLOCK.get(_block_size(q, variable_block_sizes)) is None:
        return False
    # A CTA owns an adjacent pair of query blocks.
    if variable_block_sizes.numel() % 2 != 0:
        return False
    # Metadata must be integer-typed so the wrapper's int32 conversion is value-preserving.
    if not variable_block_sizes.is_cuda or variable_block_sizes.dtype not in (torch.int32, torch.int64):
        return False
    return True


def block_sparse_attn_sm100a(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    need_lse: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass. Returns ``(out, lse)``; ``out`` has q's layout."""
    fwd = _FWD_BY_BLOCK[_block_size(q, variable_block_sizes)]
    idx = q2k_idx.to(torch.int32).contiguous()
    num = q2k_num.to(torch.int32).contiguous()
    vbs = variable_block_sizes.to(torch.int32).contiguous()
    sm_scale = 1.0 / (q.shape[-1]**0.5)
    res = fwd(q.contiguous(), k.contiguous(), v.contiguous(), None,
              idx, num, vbs, sm_scale, need_lse)
    return (res[0], res[1]) if need_lse else (res[0], None)
