# SPDX-License-Identifier: Apache-2.0
"""Rubin (sm_107a) CUDA block-sparse VSA forward, 128-token blocks.

A backend behind the same VSA op as the Triton, CuTe-DSL and ptx_blackwell paths, for Rubin devices
(compute capability 10.7). It returns ``(out, lse)`` with ``lse`` in exactly the form
``triton_block_sparse_attn_forward`` writes -- ``max(qk * qk_scale) + log2(l)``, ``[B, H, S]`` fp32.

Only 128-token sparse blocks (tile volume 128 in ``build_vsa_metadata``) have a build; anything else falls back
via ``is_supported``. The kernel picks its schedule per shape (two adjacent query blocks per CTA item, or one
query block split across the CTA's two M-tiles) on the device side; callers see one op.

``q_prescaled=True`` lets a caller that already multiplied q by ``log2(e) / sqrt(head_dim)`` skip one FMA per
score pair (~3 % at 16K+ tokens); the default takes raw q.
"""

from typing import Tuple

import torch

try:
    # The pybind symbols live on fastvideo_kernel_ops, NOT on the _C package that contains it.
    from fastvideo_kernel._C import fastvideo_kernel_ops as _C
    _FWD = getattr(_C, "block_sparse_ptx_sm107a_fwd", None)
except ImportError:  # pragma: no cover - extension not built
    _C = None
    _FWD = None
_HAS_VSA_PTX_SM107A = _FWD is not None

_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 7)}
HEAD_DIM = 128
BLOCK = 128


def _block_size(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> int:
    # A plain int: dynamic-shape torch.compile hands SymInt shapes.
    num_blocks = variable_block_sizes.numel()
    seqlen = q.shape[2]
    return 0 if num_blocks == 0 or seqlen % num_blocks else int(seqlen // num_blocks)


def is_supported(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> bool:
    """True iff this build can run these tensors; otherwise the caller uses another backend.

    Static facts only -- shapes, dtypes, arch, layout; no reads of tensor contents (no GPU sync).

    What the kernel accepts (and is tested to handle):
      * q/k/v: contiguous 4-D bf16 CUDA tensors ``[B, H, S, D]`` on an sm_107 device, head_dim 128;
        S == num_blocks * 128 with an EVEN num_blocks >= 2.
      * q2k_num: any per-row counts in [0, max_kv], non-uniform across rows included; a count-0 row gives
        exactly-zero output (and a finite LSE).
      * q2k_idx: rows need valid ascending entries (in [0, num_blocks)) below that row's count; padding past
        it (map_to_index's -1 fill) is never used as a block id.
      * variable_block_sizes: per-KV-block valid-token counts in [0, 128]; keys at or past a block's count are
        masked.
    """
    if not _HAS_VSA_PTX_SM107A or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) not in _SUPPORTED_COMPUTE_CAPABILITIES:
        return False
    if q.dtype != torch.bfloat16 or q.dim() != 4 or q.shape[-1] != HEAD_DIM:
        return False
    if not q.is_contiguous():
        return False
    if _block_size(q, variable_block_sizes) != BLOCK:
        return False
    num_blocks = variable_block_sizes.numel()
    if num_blocks < 2 or num_blocks % 2 != 0:  # a dual-Q item owns an adjacent pair of query blocks
        return False
    if not variable_block_sizes.is_cuda or variable_block_sizes.dtype not in (torch.int32, torch.int64):
        return False
    return True


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_ptx_sm107a_inference",
    mutates_args=(),
    device_types="cuda",
)
def _block_sparse_attn_ptx_sm107a_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_prescaled: bool,
) -> torch.Tensor:
    """Opaque no-LSE launch for the inference route (fake-backed, so torch.compile(fullgraph=True) captures it)."""
    sm_scale = 1.0 / (q.shape[-1]**0.5)
    res = _FWD(q, k, v, q2k_idx, q2k_num, variable_block_sizes, sm_scale, False, q_prescaled)
    return res[0]


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_ptx_sm107a_inference")
def _block_sparse_attn_ptx_sm107a_inference_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_prescaled: bool,
) -> torch.Tensor:
    # The C++ binding allocates its output with torch::empty_like(q).
    return torch.empty_like(q)


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_ptx_sm107a_from_mask_inference",
    mutates_args=(),
    device_types="cuda",
)
def _block_sparse_attn_ptx_sm107a_from_mask_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_prescaled: bool,
) -> torch.Tensor:
    """Opaque mask compaction plus no-LSE launch for inference (H3's bool block map)."""
    from fastvideo_kernel.triton_kernels.index import map_to_index

    q2k_idx, q2k_num = map_to_index(block_map)
    return _block_sparse_attn_ptx_sm107a_inference(
        q,
        k,
        v,
        q2k_idx.to(torch.int32).contiguous(),
        q2k_num.to(torch.int32).contiguous(),
        variable_block_sizes,
        q_prescaled,
    )


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_ptx_sm107a_from_mask_inference")
def _block_sparse_attn_ptx_sm107a_from_mask_inference_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_prescaled: bool,
) -> torch.Tensor:
    return torch.empty_like(q)


def block_sparse_attn_ptx_sm107a_from_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_prescaled: bool = False,
) -> Tuple[torch.Tensor, None]:
    """Inference forward from a bool block map, with compaction kept opaque."""
    out = _block_sparse_attn_ptx_sm107a_from_mask_inference(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        block_map.to(torch.bool).contiguous(),
        variable_block_sizes.to(torch.int32).contiguous(),
        q_prescaled,
    )
    return out, None


def block_sparse_attn_ptx_sm107a(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    need_lse: bool = True,
    q_prescaled: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Forward pass. Returns ``(out, lse)``; ``out`` has q's layout, ``lse`` is None when not asked for."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    idx = q2k_idx.to(torch.int32).contiguous()
    num = q2k_num.to(torch.int32).contiguous()
    vbs = variable_block_sizes.to(torch.int32).contiguous()

    if not need_lse:
        # The production inference path: the custom op keeps the raw pybind launch opaque to Dynamo.
        return _block_sparse_attn_ptx_sm107a_inference(q, k, v, idx, num, vbs, q_prescaled), None

    sm_scale = 1.0 / (q.shape[-1]**0.5)
    res = _FWD(q, k, v, idx, num, vbs, sm_scale, True, q_prescaled)
    return res[0], res[1]
