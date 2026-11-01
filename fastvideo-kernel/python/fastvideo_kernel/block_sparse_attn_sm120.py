# SPDX-License-Identifier: Apache-2.0
"""Opt-in SM120 BF16 tile-64 sparse attention forward (eager inference only).

K/V async copies overlap QK, softmax and PV using distinct shared buffers.
The sparse graph and its index order are unchanged. The MMA reduction layout
can change BF16 rounding relative to native Triton; this is not bit-exact.
"""

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp


@gluon.jit
def _issue_overlap_tile(first, second, pointers):
    # The caller owns publication and buffer-reuse synchronization.
    cp.async_copy_global_to_shared(first, pointers)
    cp.async_copy_global_to_shared(second, pointers + 64)
    cp.commit_group()


@gluon.jit
def _sm120_sparse_async_overlap2_fwd(
    Q, K, V, Indices, Counts, Sizes, Out, Lse,
    Q_TILES: gl.constexpr, KV_TILES: gl.constexpr,
    CAPACITY: gl.constexpr, SCALE: gl.constexpr,
):
    WARPS: gl.constexpr = 4
    mma: gl.constexpr = gl.NVMMADistributedLayout([2, 0], [4, WARPS // 4], [16, 8])
    lhs: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [WARPS, 1], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, mma)
    col_layout: gl.constexpr = gl.SliceLayout(0, mma)
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    # Two 16-KiB tiles, with distinct allocation roots for each channel half.
    k_first = gl.allocate_shared_memory(gl.bfloat16, [64, 64], shared_layout)
    k_second = gl.allocate_shared_memory(gl.bfloat16, [64, 64], shared_layout)
    v_first = gl.allocate_shared_memory(gl.bfloat16, [64, 64], shared_layout)
    v_second = gl.allocate_shared_memory(gl.bfloat16, [64, 64], shared_layout)
    query_tile = gl.program_id(0)
    head = gl.program_id(1).to(gl.int64)
    n_query: gl.constexpr = Q_TILES * 64
    n_key: gl.constexpr = KV_TILES * 64
    metadata_row = head * Q_TILES + query_tile
    count = gl.load(Counts + metadata_row)
    index_base = Indices + metadata_row * CAPACITY
    rows = gl.arange(0, 64, layout=gl.SliceLayout(1, load_layout))
    channels = gl.arange(0, 64, layout=gl.SliceLayout(0, load_layout))
    offsets = rows[:, None] * 128 + channels[None, :]
    q_base = Q + head * n_query * 128 + query_tile * 8192
    q_first = gl.convert_layout(gl.load(q_base + offsets), lhs)
    q_second = gl.convert_layout(gl.load(q_base + offsets + 64), lhs)
    k_base = K + head * n_key * 128
    v_base = V + head * n_key * 128
    columns = gl.arange(0, 64, layout=col_layout)
    maximum = gl.full((64,), -float("inf"), gl.float32, row_layout)
    denominator = gl.full((64,), 1.0, gl.float32, row_layout)
    acc_first = gl.full((64, 64), 0.0, gl.float32, mma)
    acc_second = gl.full((64, 64), 0.0, gl.float32, mma)
    qk_scale: gl.constexpr = SCALE * 1.44269504
    tile = gl.load(index_base, mask=count > 0, other=0)
    size = gl.load(Sizes + tile, mask=count > 0, other=0)
    if count > 0:
        _issue_overlap_tile(k_first, k_second, k_base + tile * 8192 + offsets)

    for index in range(count):
        # Publish K and finish all prior V reads before issuing the next V copy.
        cp.wait_group(0)
        gl.barrier()
        # Masked cp.async zero-fills the invalid value rows.
        valid_values = rows[:, None] < size
        value_pointers = v_base + tile * 8192 + offsets
        cp.async_copy_global_to_shared(v_first, value_pointers, mask=valid_values)
        cp.async_copy_global_to_shared(v_second, value_pointers + 64, mask=valid_values)
        cp.commit_group()
        has_next = index + 1 < count
        next_tile = gl.load(index_base + index + 1, mask=has_next, other=0)
        next_size = gl.load(Sizes + next_tile, mask=has_next, other=0)
        scores = mma_v2(q_first, k_first.permute((1, 0)).load(rhs),
                        gl.full((64, 64), 0.0, gl.float32, mma))
        scores = mma_v2(q_second, k_second.permute((1, 0)).load(rhs), scores)
        # All warps must finish reading K before its buffer holds the next K.
        gl.barrier()
        if has_next:
            _issue_overlap_tile(k_first, k_second, k_base + next_tile * 8192 + offsets)

        scores = gl.where(columns[None, :] < size, scores, -float("inf"))
        next_maximum = gl.maximum(maximum, gl.max(scores, 1) * qk_scale)
        # A selected zero-valid tile must not introduce exp2(-inf - -inf).
        normalizer = gl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        probabilities = gl.exp2(scores * qk_scale - normalizer[:, None])
        alpha = gl.exp2(maximum - normalizer)
        denominator = denominator * alpha + gl.sum(probabilities, 1)
        acc_first = acc_first * alpha[:, None]
        acc_second = acc_second * alpha[:, None]
        p = gl.convert_layout(probabilities.to(gl.bfloat16), lhs)
        if has_next:
            # V was committed before the next K; only that newer K may remain pending.
            cp.wait_group(1)
        else:
            cp.wait_group(0)
        gl.barrier()
        acc_first = mma_v2(p, v_first.load(rhs), acc_first)
        acc_second = mma_v2(p, v_second.load(rhs), acc_second)
        maximum = next_maximum
        gl.barrier()
        tile = next_tile
        size = next_size

    cp.wait_group(0)
    denominator = gl.where(denominator > 0, denominator, 1.0)
    out_base = Out + head * n_query * 128 + query_tile * 8192
    result_first = gl.convert_layout((acc_first / denominator[:, None]).to(gl.bfloat16), load_layout)
    gl.store(out_base + offsets, result_first)
    result_second = gl.convert_layout((acc_second / denominator[:, None]).to(gl.bfloat16), load_layout)
    gl.store(out_base + offsets + 64, result_second)
    lse_rows = query_tile * 64 + gl.arange(0, 64, layout=row_layout)
    gl.store(Lse + head * n_query + lse_rows, maximum + gl.log2(denominator))


def block_sparse_attn_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward on SM120 with contiguous BF16 BHSD inputs and head dim 128.

    Sequences are padded to multiples of 64; Q and KV lengths may differ.
    Indices have shape [B, H, Q_tiles, capacity], counts [B, H, Q_tiles],
    and shared KV valid sizes [KV_tiles]. Integer metadata must be contiguous,
    on the same device, with counts in [0, capacity], selected indices in
    [0, KV_tiles), and sizes in [0, 64]. Unused index slots are not read.
    These value invariants are the caller's responsibility, as for native
    Triton; checking them here would synchronize the CPU on every layer.
    Invalid KV rows are masked, including V padding. All padded query rows
    are computed; callers discard them when untiling. Empty rows return zero
    output and -inf log2-LSE. No backward or torch.compile support is provided.
    """
    if torch.compiler.is_compiling():
        raise RuntimeError("SM120 sparse attention requires eager execution; disable DiT compilation")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("SM120 sparse attention expects BHSD tensors")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("SM120 sparse attention requires a CUDA device with compute capability 12.0")
    if any(t.device != q.device for t in (k, v, q2k_idx, q2k_num, variable_block_sizes)):
        raise ValueError("SM120 sparse attention inputs must share a device")
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (q, k, v)):
        raise ValueError("SM120 sparse attention requires contiguous BF16 Q/K/V")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("SM120 sparse attention is forward-only; disable gradient tracking")
    batch, heads, nq, dim = q.shape
    nk = k.shape[2]
    if (min(batch, heads, nq, nk) < 1 or dim != 128 or k.shape != v.shape
            or k.shape[:2] != (batch, heads) or k.shape[3] != dim or nq % 64 or nk % 64):
        raise ValueError("SM120 sparse attention requires matching B/H, D=128 and positive tile-64 lengths")
    if max(nq, nk) * dim >= 2**31:
        raise ValueError("SM120 sparse attention per-head offsets must fit int32")
    if (q2k_idx.ndim != 4 or q2k_idx.shape[:3] != (batch, heads, nq // 64)
            or q2k_idx.shape[3] < 1 or q2k_num.shape != q2k_idx.shape[:3]
            or variable_block_sizes.shape != (nk // 64,)):
        raise ValueError("SM120 sparse attention metadata does not match the tiled Q/K/V shapes")
    for tensor in (q2k_idx, q2k_num, variable_block_sizes):
        if tensor.dtype not in (torch.int32, torch.int64) or not tensor.is_contiguous():
            raise ValueError("SM120 sparse attention requires contiguous int32/int64 metadata")
    out = torch.empty_like(q)
    lse = torch.empty((batch, heads, nq), device=q.device, dtype=torch.float32)
    with torch.cuda.device(q.device):
        _sm120_sparse_async_overlap2_fwd[(nq // 64, batch * heads)](
            q, k, v, q2k_idx, q2k_num, variable_block_sizes, out, lse,
            Q_TILES=nq // 64, KV_TILES=nk // 64, CAPACITY=q2k_idx.shape[3],
            SCALE=dim**-0.5, num_warps=4, num_stages=1,
        )
    return out, lse


