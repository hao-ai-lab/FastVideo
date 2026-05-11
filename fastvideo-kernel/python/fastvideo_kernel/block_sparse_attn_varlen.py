"""Variable-length block-sparse attention via sequence packing.

Packs multiple variable-length sequences into a single [1, H, T_total, D]
tensor and delegates to the existing block_sparse_attn_from_indices kernel
in a single launch. No kernel modifications required.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .block_sparse_attn import block_sparse_attn_from_indices

BLOCK_SIZE = 64


def _scatter_to_padded(
    src: torch.Tensor,
    block_sizes: torch.Tensor,
    block_size: int,
    dst: torch.Tensor,
    dst_offset: int,
    src_start: int,
    src_end: int,
) -> None:
    """Copy tokens from a flat source into block-aligned positions in dst.

    Each block occupies exactly `block_size` slots in dst.  The first
    `block_sizes[b]` slots of block *b* receive real tokens; the remainder
    stays zero (padding the kernel expects).

    src:         [total_tokens, H, D]
    dst:         [1, H, total_padded, D]
    block_sizes: [num_blocks] int32, actual token count per block.
    """
    src_pos = src_start
    dst_pos = dst_offset
    for b in range(block_sizes.numel()):
        actual = int(block_sizes[b].item())
        actual = min(actual, src_end - src_pos)
        if actual > 0:
            dst[:, :, dst_pos:dst_pos + actual, :] = (
                src[src_pos:src_pos + actual].transpose(0, 1).unsqueeze(0)
            )
        src_pos += actual
        dst_pos += block_size


def _gather_from_padded(
    src: torch.Tensor,
    block_sizes: torch.Tensor,
    block_size: int,
    dst: torch.Tensor,
    src_offset: int,
    dst_start: int,
    dst_end: int,
) -> None:
    """Inverse of _scatter_to_padded: extract real tokens from padded blocks.

    src: [1, H, total_padded, D]
    dst: [total_tokens, H, D]
    """
    src_pos = src_offset
    dst_pos = dst_start
    for b in range(block_sizes.numel()):
        actual = int(block_sizes[b].item())
        actual = min(actual, dst_end - dst_pos)
        if actual > 0:
            dst[dst_pos:dst_pos + actual] = (
                src[0, :, src_pos:src_pos + actual, :].transpose(0, 1)
            )
        dst_pos += actual
        src_pos += block_size


def block_sparse_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q2k_idx_list: Sequence[torch.Tensor],
    q2k_num_list: Sequence[torch.Tensor],
    variable_block_sizes_list: Sequence[torch.Tensor],
    q_variable_block_sizes_list: Sequence[torch.Tensor] | None = None,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Block-sparse attention over packed variable-length sequences.

    Args:
        q: [total_q_tokens, H, D] packed query tensor.
        k: [total_kv_tokens, H, D] packed key tensor.
        v: [total_kv_tokens, H, D] packed value tensor.
        cu_seqlens_q: [N+1] int32, cumulative Q token offsets.
        cu_seqlens_kv: [N+1] int32, cumulative KV token offsets.
        q2k_idx_list: Per-sequence q2k_idx tensors, each [1, H, Nq_i, Mk].
        q2k_num_list: Per-sequence q2k_num tensors, each [1, H, Nq_i].
        variable_block_sizes_list: Per-sequence KV block sizes, each [Nkv_i].
        q_variable_block_sizes_list: Per-sequence Q block sizes, each [Nq_i].
            If None, each Q block is assumed to be exactly `block_size` tokens.
        block_size: Attention block size (default 64).

    Returns:
        out: [total_q_tokens, H, D] packed output tensor.
    """
    device = q.device
    dtype = q.dtype
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_seqs = cu_seqlens_q.shape[0] - 1

    cu_q = cu_seqlens_q.cpu().tolist()
    cu_kv = cu_seqlens_kv.cpu().tolist()

    padded_q_lens = []
    padded_kv_lens = []
    q_block_offsets = [0]
    kv_block_offsets = [0]
    q_vbs_resolved = []

    for i in range(num_seqs):
        n_q_blocks = q2k_num_list[i].shape[-1]
        n_kv_blocks = variable_block_sizes_list[i].numel()
        padded_q_lens.append(n_q_blocks * block_size)
        padded_kv_lens.append(n_kv_blocks * block_size)
        q_block_offsets.append(q_block_offsets[-1] + n_q_blocks)
        kv_block_offsets.append(kv_block_offsets[-1] + n_kv_blocks)

        if q_variable_block_sizes_list is not None:
            q_vbs_resolved.append(q_variable_block_sizes_list[i])
        else:
            q_vbs_resolved.append(
                torch.full((n_q_blocks,), block_size, dtype=torch.int32, device=device)
            )

    total_padded_q = sum(padded_q_lens)
    total_padded_kv = sum(padded_kv_lens)

    q_packed = torch.zeros(1, num_heads, total_padded_q, head_dim, device=device, dtype=dtype)
    k_packed = torch.zeros(1, num_heads, total_padded_kv, head_dim, device=device, dtype=dtype)
    v_packed = torch.zeros(1, num_heads, total_padded_kv, head_dim, device=device, dtype=dtype)

    q_offset = 0
    kv_offset = 0
    for i in range(num_seqs):
        _scatter_to_padded(
            q, q_vbs_resolved[i], block_size,
            q_packed, q_offset, cu_q[i], cu_q[i + 1],
        )
        _scatter_to_padded(
            k, variable_block_sizes_list[i], block_size,
            k_packed, kv_offset, cu_kv[i], cu_kv[i + 1],
        )
        _scatter_to_padded(
            v, variable_block_sizes_list[i], block_size,
            v_packed, kv_offset, cu_kv[i], cu_kv[i + 1],
        )
        q_offset += padded_q_lens[i]
        kv_offset += padded_kv_lens[i]

    total_q_blocks = q_block_offsets[-1]
    max_kv_per_q = max(t.shape[-1] for t in q2k_idx_list)

    global_q2k_idx = torch.zeros(
        1, num_heads, total_q_blocks, max_kv_per_q,
        dtype=torch.int32, device=device,
    )
    global_q2k_num = torch.zeros(
        1, num_heads, total_q_blocks,
        dtype=torch.int32, device=device,
    )
    global_vbs_parts = []

    for i in range(num_seqs):
        qb_start = q_block_offsets[i]
        qb_end = q_block_offsets[i + 1]
        n_q_blocks = qb_end - qb_start
        kv_offset_blocks = kv_block_offsets[i]

        idx = q2k_idx_list[i]
        num = q2k_num_list[i]
        vbs = variable_block_sizes_list[i]

        mk = idx.shape[-1]
        global_q2k_idx[:, :, qb_start:qb_end, :mk] = idx[:, :, :n_q_blocks, :] + kv_offset_blocks
        global_q2k_num[:, :, qb_start:qb_end] = num[:, :, :n_q_blocks]
        global_vbs_parts.append(vbs)

    global_vbs = torch.cat(global_vbs_parts, dim=0).to(torch.int32).contiguous()

    out_packed, _ = block_sparse_attn_from_indices(
        q_packed, k_packed, v_packed,
        global_q2k_idx, global_q2k_num, global_vbs,
    )

    out = torch.zeros(cu_q[-1], num_heads, head_dim, device=device, dtype=dtype)
    q_offset = 0
    for i in range(num_seqs):
        _gather_from_padded(
            out_packed, q_vbs_resolved[i], block_size,
            out, q_offset, cu_q[i], cu_q[i + 1],
        )
        q_offset += padded_q_lens[i]

    return out
