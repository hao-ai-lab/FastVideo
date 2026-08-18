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
    """True iff this build can run these tensors; otherwise the caller uses Triton."""
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
    # A fully empty block would give an all -inf row; FastVideo's tiling does not produce one,
    # but the kernel assumes it and the check is a single reduction.
    if int(variable_block_sizes.min()) < 1:
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
