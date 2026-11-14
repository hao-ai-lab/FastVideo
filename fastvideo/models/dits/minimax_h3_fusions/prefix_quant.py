# SPDX-License-Identifier: Apache-2.0
"""FP8 prefix-key preparation fused with valid-key/value gathering."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefix_kv(K, V, IDX, KO, VO, KB: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr,
               VB: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr,
               B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    col = tl.arange(0, D)
    batch = row // (N * H)
    token = row // H % N
    head = row % H
    source = tl.load(IDX + token, row < B * N * H, other=0).to(tl.int64)
    mask = row[:, None] < B * N * H
    k = tl.load(K + batch[:, None] * KB + source[:, None] * KS + head[:, None] * KH + col[None, :],
                mask, other=0).to(tl.float32)
    scale = tl.max(tl.abs(k), 1) / 448.0
    inv = tl.where(scale == 0, 0, 1.0 / scale)
    rounded = tl.minimum(tl.maximum(k * inv[:, None], -448.0), 448.0).to(tl.float8e4nv).to(tl.float32)
    tl.store(KO + row[:, None] * D + col[None, :], rounded * scale[:, None], mask)
    v = tl.load(V + batch[:, None] * VB + source[:, None] * VS + head[:, None] * VH + col[None, :],
                mask, other=0)
    tl.store(VO + row[:, None] * D + col[None, :], v, mask)


def quantize_gather_prefix_kv(
    k: torch.Tensor, v: torch.Tensor, index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the existing per-head FP8 round trip to K and gather valid K/V rows.

    K remains BF16 at the SDPA boundary. The output keeps the existing BSHD
    layout, and each head's amax is independent of the other token rows.
    """
    if (not k.is_cuda or k.dtype != torch.bfloat16 or v.dtype != k.dtype or v.device != k.device
            or k.shape != v.shape or k.ndim != 4 or k.shape[-1] != 128
            or k.stride(-1) != 1 or v.stride(-1) != 1):
        raise ValueError("Prefix quantization requires CUDA BF16 BSHD tensors with contiguous 128-wide heads")
    if index.ndim != 1 or index.device != k.device or index.dtype not in (torch.int32, torch.int64):
        raise ValueError("Prefix quantization requires a CUDA integer row index")
    if torch.is_grad_enabled():
        raise RuntimeError("Prefix quantization is inference-only")
    batch, _, heads, dim = k.shape
    rows = index.numel()
    ko = torch.empty((batch, rows, heads, dim), device=k.device, dtype=k.dtype)
    vo = torch.empty_like(ko)
    if rows:
        _prefix_kv[(triton.cdiv(batch * rows * heads, 4),)](
            k, v, index, ko, vo, *k.stride()[:3], *v.stride()[:3], batch, rows, heads, dim, 4, num_warps=4,
        )
    return ko, vo
