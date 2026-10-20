# SPDX-License-Identifier: Apache-2.0
"""GB10 (DGX Spark, sm_121a) CUDA block-sparse VSA forward, 128-token blocks.

The GB10 counterpart of ``block_sparse_attn_ptx_blackwell``'s 128-token forward, which is a
tcgen05/tensor-memory kernel the GB10 cannot run. This one is an mma.sync + TMA kernel built for
sm_121a (``csrc/attention/block_sparse_kernel_blk128_sm121.cuh``). It computes the same output
for the same inputs, and nothing else: it is inference-only and returns no LSE, so it has no
backward and is never reached by grad-tracking calls.

It is a separate module, not another capability in ``block_sparse_attn_ptx_blackwell``, because
that module's ``is_supported`` also gates ``block_sparse_attn_from_indices``'s autograd op, which
always asks for the LSE.
"""

from typing import Tuple

import torch

try:
    # The pybind symbols live on fastvideo_kernel_ops, not on the _C namespace package.
    from fastvideo_kernel._C import fastvideo_kernel_ops as _C
    _FWD = getattr(_C, "block_sparse_sm121_blk128_fwd", None)
except ImportError:  # pragma: no cover - extension not built
    _C = None
    _FWD = None

_HAS_VSA_SM121 = _FWD is not None
_SUPPORTED_COMPUTE_CAPABILITIES = {(12, 1)}
BLOCK = 128
HEAD_DIM = 128


def is_supported(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> bool:
    """True iff this build can run these tensors on this device; otherwise the caller falls back.

    Static facts only (shapes, dtypes, device, layout), so it never synchronizes with the GPU:
      * the extension carries the sm_121a kernel and q is on a GB10 (compute capability 12.1);
      * q is a contiguous 4-D bf16 BHSD tensor with head_dim 128 and seqlen == num_blocks * 128,
        num_blocks = variable_block_sizes.numel(); any num_blocks, odd included;
      * variable_block_sizes is an integer CUDA tensor.
    What the kernel accepts beyond that, and is tested to handle: per-row counts q2k_num in
    [0, max_kv], non-uniform across rows (a row with count 0 gets exactly-zero output); q2k_idx
    entries are only read below each row's count (map_to_index's -1 padding is never read); any
    variable_block_sizes in [0, 128] (keys at or past a block's count are masked, and query rows
    past it are written zero).
    """
    if not _HAS_VSA_SM121 or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) not in _SUPPORTED_COMPUTE_CAPABILITIES:
        return False
    if q.dtype != torch.bfloat16 or q.dim() != 4 or q.shape[-1] != HEAD_DIM or not q.is_contiguous():
        return False
    num_blocks = variable_block_sizes.numel()
    if num_blocks == 0 or q.shape[2] != num_blocks * BLOCK:
        return False
    if not variable_block_sizes.is_cuda or variable_block_sizes.dtype not in (torch.int32, torch.int64):
        return False
    return True


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_sm121_inference",
    mutates_args=(),
    device_types="cuda",
)
def _block_sparse_attn_sm121_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Opaque launch, so that Dynamo does not descend into the raw pybind function."""
    return _FWD(q, k, v, q2k_idx, q2k_num, variable_block_sizes)[0]


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_sm121_inference")
def _block_sparse_attn_sm121_inference_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    # The C++ binding allocates its output with torch::empty_like(q).
    return torch.empty_like(q)


def block_sparse_attn_sm121(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, None]:
    """Forward pass, inference only. Returns ``(out, None)``; ``out`` has q's [B, H, S, 128] layout.

    The signature matches ``block_sparse_attn_ptx_blackwell(..., need_lse=False)``: q2k_idx is
    [B, H, num_blocks, max_kv] and q2k_num [B, H, num_blocks], as ``map_to_index`` returns them.
    """
    if _FWD is None:
        raise RuntimeError("fastvideo_kernel was built without the sm_121a block-sparse forward "
                           "(build with TORCH_CUDA_ARCH_LIST containing 12.1a and CUDA >= 13)")
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    idx = q2k_idx.to(torch.int32).contiguous()
    num = q2k_num.to(torch.int32).contiguous()
    vbs = variable_block_sizes.to(torch.int32).contiguous()
    return _block_sparse_attn_sm121_inference(q, k, v, idx, num, vbs), None
