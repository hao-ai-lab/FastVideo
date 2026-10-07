# Modified from the original SageATtention3 code
"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda

# Centralized block size configuration for sageattn_blackwell kernels
# These should match the values in fastvideo/attention/backends/sageattn/blackwell/block_config.h
BLOCK_M = 128  # Block size for M dimension (query sequence length)
BLOCK_N = 128  # Block size for N dimension (key/value sequence length)


@triton.jit
def group_mean_kernel(q_ptr, q_out_ptr, qm_out_ptr, B, H, L, D: tl.constexpr, stride_qb, stride_qh, stride_ql,
                      stride_qd, stride_qmb, stride_qmh, stride_qml, stride_qmd, GROUP_SIZE: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)

    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)

    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0,
                                                                                                 D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets)

    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE

    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    B, H, L, D = q.shape
    GROUP_SIZE = BLOCK_M
    num_groups = L // GROUP_SIZE

    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)

    grid = (B, H, num_groups)

    group_mean_kernel[grid](q,
                            q_out,
                            qm,
                            B,
                            H,
                            L,
                            D,
                            q.stride(0),
                            q.stride(1),
                            q.stride(2),
                            q.stride(3),
                            qm.stride(0),
                            qm.stride(1),
                            qm.stride(2),
                            qm.stride(3),
                            GROUP_SIZE=GROUP_SIZE)
    return q_out, qm


def preprocess_qkv(q: torch.Tensor,
                   k: torch.Tensor,
                   v: torch.Tensor,
                   per_block_mean: bool = True,
                   enable_smoothing_q: bool = False,
                   enable_smoothing_k: bool = False):

    def pad_to_block_size(x):
        L = x.size(2)
        pad_len = (BLOCK_M - L % BLOCK_M) % BLOCK_M
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    if enable_smoothing_k:
        k -= k.mean(dim=-2, keepdim=True)
    q, k, v = map(lambda x: pad_to_block_size(x), [q, k, v])
    if per_block_mean and enable_smoothing_q:
        q, qm = triton_group_mean(q)
    elif enable_smoothing_q:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    if enable_smoothing_q:
        delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    else:  # used to disable q smoothing
        delta_s = _zero_delta_s(q.shape[0], q.shape[1], k.shape[2], q.device)

    return q, k, v, delta_s


_ZERO_DELTA_S: dict = {}


def _zero_delta_s(batch: int, heads: int, kv_len: int, device: torch.device) -> torch.Tensor:
    """Cached all-zero delta_s for unsmoothed Q, read with per_block_mean=False.

    The kernel reads delta_s as a contiguous [B, H, rows, KL] tensor with one
    row per query block when per_block_mean is set and a single row otherwise.
    With Q smoothing off every row is zero, so one shared row replaces the
    [B, H, L/128, KL] tensor that was allocated and zero-filled on every call
    (9.5 GB at 73k tokens, whose int32 batch stride also broke the TMA
    descriptor). The kernel only reads it.
    """
    key = (batch, heads, kv_len, device)
    zeros = _ZERO_DELTA_S.get(key)
    if zeros is None:
        zeros = torch.zeros((batch, heads, 1, kv_len), device=device, dtype=torch.float32)
        _ZERO_DELTA_S[key] = zeros
    return zeros


def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def blockscaled_fp4_attn(qlist: Tuple,
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False,
                         per_block_mean: bool = True,
                         is_bf16: bool = True,
                         single_level_p_quant: bool = False,
                         sm_scale: float | None = None):
    softmax_scale = sm_scale if sm_scale is not None else (qlist[0].shape[-1] * 2)**(-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None,
                            softmax_scale, is_causal, per_block_mean, is_bf16, single_level_p_quant)


def blockscaled_fp4_attn_sparse(qlist: Tuple,
                                klist: Tuple,
                                vlist: Tuple,
                                delta_s: torch.Tensor,
                                KL: int,
                                q2k_idx: torch.Tensor,
                                q2k_num: torch.Tensor,
                                kv_valid: torch.Tensor | None = None,
                                q2k_quad: torch.Tensor | None = None,
                                per_block_mean: bool = True,
                                is_bf16: bool = True,
                                single_level_p_quant: bool = False,
                                sm_scale: float | None = None):
    softmax_scale = sm_scale if sm_scale is not None else (qlist[0].shape[-1] * 2)**(-0.5)
    return fp4attn_cuda.fwd_sparse(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None,
                                   softmax_scale, per_block_mean, is_bf16, single_level_p_quant, q2k_idx, q2k_num,
                                   kv_valid, q2k_quad)


HALF_N = BLOCK_N // 2


def vsa_tile_mask_to_fp4_blocks(
    tile_mask: torch.Tensor,
    tile_tokens: int,
    tile_valid: torch.Tensor | None = None,
    validate: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert a VSA tile mask into the FP4 kernel's block lists.

    ``tile_mask`` is a ``[B, H, T, T]`` bool mask over VSA tiles of
    ``tile_tokens`` tokens (query tile, key tile); ``tile_tokens`` is 64 or a
    multiple of 128. ``tile_valid`` holds each tile's valid token count (valid
    tokens first, padding last, as VSA tiling lays them out). The kernel works
    on 128x128 blocks, so 64-token tiles are paired: a block is listed when any
    of its four 64x64 quadrants is selected, and ``q2k_quad`` carries which.

    Lists run in descending block order because the kernel visits them last
    entry first: block 0 (the first prefix tile, which VSA-H3's exempt mode
    gives every query) is then visited first, so every query row starts from a
    finite running max. ``validate=True`` checks that (host sync).

    Returns ``(q2k_idx [B, H, M, N], q2k_num [B, H, M], kv_valid [2N],
    q2k_quad [B, H, M, N] | None)``; the queries cover ``M * 128`` padded rows.
    """
    if tile_tokens % HALF_N or (tile_tokens > HALF_N and tile_tokens % BLOCK_N):
        raise ValueError(f"tile_tokens={tile_tokens} must be {HALF_N} or a multiple of {BLOCK_N}")
    batch, heads, n_tiles, _ = tile_mask.shape
    device = tile_mask.device
    halves_per_tile = tile_tokens // HALF_N
    half_mask = tile_mask
    if halves_per_tile > 1:
        half_mask = half_mask.repeat_interleave(halves_per_tile, dim=2).repeat_interleave(halves_per_tile, dim=3)
    n_halves = n_tiles * halves_per_tile
    if tile_valid is None:
        half_valid = torch.full((n_halves, ), HALF_N, device=device, dtype=torch.int32)
    else:
        offsets = torch.arange(halves_per_tile, device=device, dtype=torch.int32) * HALF_N
        half_valid = (tile_valid.to(torch.int32)[:, None] - offsets[None, :]).clamp(0, HALF_N).reshape(-1)
    if n_halves % 2:
        # Pad to whole 128-blocks: the extra key half is empty, the extra query
        # half (discarded output) attends block 0 only.
        half_mask = F.pad(half_mask, (0, 1, 0, 1), value=False)
        half_mask[:, :, -1, 0] = True
        half_valid = F.pad(half_valid, (0, 1), value=0)
        n_halves += 1
    half_mask = half_mask & (half_valid > 0)[None, None, None, :]
    n_blocks = n_halves // 2
    quads = half_mask.view(batch, heads, n_blocks, 2, n_blocks, 2)
    weights = torch.tensor([[1, 2], [4, 8]], device=device, dtype=torch.uint8)  # [row_half, col_half]
    quad = (quads.to(torch.uint8) * weights[None, None, None, :, None, :]).sum(dim=(3, 5), dtype=torch.uint8)
    block_mask = quad != 0
    # Descending compaction without a sort: position = running count from the right.
    rev = block_mask.flip(-1)
    pos = rev.cumsum(-1, dtype=torch.int32) - 1
    q2k_num = (pos[..., -1] + 1).contiguous()
    cols = torch.arange(n_blocks - 1, -1, -1, device=device, dtype=torch.int32).expand_as(pos)
    slot = torch.where(rev, pos, torch.full_like(pos, n_blocks)).long()
    q2k_idx = torch.zeros((batch, heads, n_blocks, n_blocks + 1), device=device, dtype=torch.int32)
    q2k_idx.scatter_(-1, slot, cols)
    q2k_idx = q2k_idx[..., :n_blocks].contiguous()
    kv_valid = half_valid.contiguous()
    q2k_quad = None
    if halves_per_tile == 1:
        q2k_quad = torch.zeros((batch, heads, n_blocks, n_blocks + 1), device=device, dtype=torch.uint8)
        q2k_quad.scatter_(-1, slot, quad.flip(-1))
        q2k_quad = q2k_quad[..., :n_blocks].contiguous()
    if validate:
        if int(q2k_num.min()) < 1:
            raise ValueError("every query block must attend to at least one non-empty KV block")
        last = (q2k_num - 1).long().unsqueeze(-1)
        first_block = q2k_idx.gather(-1, last.int().long()).squeeze(-1).long()
        first_quad = (q2k_quad.gather(-1, last).squeeze(-1).int() if q2k_quad is not None else
                      torch.full_like(first_block, 15, dtype=torch.int32))
        v0 = (kv_valid[2 * first_block] > 0).int()
        v1 = (kv_valid[2 * first_block + 1] > 0).int()
        row0 = ((first_quad & 1).bool() & v0.bool()) | ((first_quad & 2).bool() & v1.bool())
        row1 = ((first_quad & 4).bool() & v0.bool()) | ((first_quad & 8).bool() & v1.bool())
        if not bool((row0 & row1).all()):
            raise ValueError("the first block each query block visits must give both 64-row halves a valid key")
    return q2k_idx, q2k_num, kv_valid, q2k_quad


def check_sparse_block_lists(q2k_idx: torch.Tensor, q2k_num: torch.Tensor, kv_len: int) -> None:
    """Reject block lists the sparse kernel would read out of bounds (host sync).

    Each row needs ``1 <= q2k_num <= q2k_idx.size(-1)`` and every listed index
    in ``[0, ceil(kv_len / BLOCK_N))``; a zero count or an out-of-range index
    makes the kernel load outside its index row or the KV tensors.
    """
    num_kv_blocks = -(-kv_len // BLOCK_N)
    if int(q2k_num.min()) < 1 or int(q2k_num.max()) > q2k_idx.size(-1):
        raise ValueError(f"q2k_num must be in [1, {q2k_idx.size(-1)}]")
    listed = torch.arange(q2k_idx.size(-1), device=q2k_idx.device) < q2k_num.unsqueeze(-1)
    idx = q2k_idx[listed]
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= num_kv_blocks):
        raise ValueError(f"q2k_idx entries must be in [0, {num_kv_blocks})")


def sageattn_blackwell_sparse(q,
                              k,
                              v,
                              q2k_idx: torch.Tensor,
                              q2k_num: torch.Tensor,
                              kv_valid: torch.Tensor | None = None,
                              q2k_quad: torch.Tensor | None = None,
                              per_block_mean=True,
                              single_level_p_quant=True,
                              sm_scale: float | None = None,
                              validate: bool = True):
    """Block-sparse SageAttention3 FP4 forward (non-causal).

    Query block ``m`` (``BLOCK_M`` rows) of each (batch, head) attends only to
    the ``BLOCK_N``-token KV blocks in ``q2k_idx[b, h, m, :q2k_num[b, h, m]]``,
    restricted to the quadrants in ``q2k_quad`` when given; see
    :func:`vsa_tile_mask_to_fp4_blocks`. Q/K/V are ``[B, H, L, D]``.
    Block lists are checked with :func:`check_sparse_block_lists` (a host
    sync) unless ``validate=False``; pass that only for lists built by
    :func:`vsa_tile_mask_to_fp4_blocks`, which are in range by construction.
    """
    QL = q.size(2)
    KL = k.size(2)
    if validate:
        check_sparse_block_lists(q2k_idx, q2k_num, KL)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    per_block_mean = delta_s.shape[2] > 1
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    o_fp4 = blockscaled_fp4_attn_sparse(qlist_from_cuda, klist_from_cuda, vlist_from_cuda, delta_s, KL, q2k_idx,
                                        q2k_num, kv_valid, q2k_quad, per_block_mean, is_bf16, single_level_p_quant,
                                        sm_scale)[0][:, :, :QL, :].contiguous()
    return o_fp4


def sageattn_blackwell_sparse_bshd(q,
                                   k,
                                   v,
                                   q2k_idx: torch.Tensor,
                                   q2k_num: torch.Tensor,
                                   kv_valid: torch.Tensor | None = None,
                                   q2k_quad: torch.Tensor | None = None,
                                   single_level_p_quant=True,
                                   sm_scale: float | None = None,
                                   validate: bool = True) -> torch.Tensor:
    """:func:`sageattn_blackwell_sparse` for ``[B, L, H, D]`` inputs, without copies.

    The FP4 quantizers read strided input, so the sequence-major tensors a
    linear produces are quantized in place of a transpose + pad. ``L`` must be
    a multiple of ``BLOCK_M`` (callers allocate the padding) and Q is
    unsmoothed. Returns ``[B, H, L, D]``.
    """
    batch, seq_len, heads, _ = q.shape
    if seq_len % BLOCK_M:
        raise ValueError(f"sequence length {seq_len} must be a multiple of {BLOCK_M}")
    if validate:
        check_sparse_block_lists(q2k_idx, q2k_num, seq_len)
    qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
    delta_s = _zero_delta_s(batch, heads, seq_len, q.device)
    return blockscaled_fp4_attn_sparse(scale_and_quant_fp4(qh), scale_and_quant_fp4_permute(kh),
                                       scale_and_quant_fp4_transpose(vh), delta_s, seq_len, q2k_idx, q2k_num, kv_valid,
                                       q2k_quad, False, q.dtype == torch.bfloat16, single_level_p_quant, sm_scale)[0]


def sageattn_blackwell(q,
                       k,
                       v,
                       attn_mask=None,
                       is_causal=False,
                       per_block_mean=True,
                       single_level_p_quant=True,
                       sm_scale: float | None = None,
                       **kwargs):
    """
    SageAttention3 Blackwell kernel for FP4 attention.
    
    Args:
        q: Query tensor [B, H, L, D]
        k: Key tensor [B, H, L, D]
        v: Value tensor [B, H, L, D]
        attn_mask: Attention mask (not used)
        is_causal: Whether to use causal masking
        per_block_mean: Whether to use per-block mean for Q smoothing
        single_level_p_quant: If True, use single-level quantization: s_P2, P̂_2 = φ(P̃) directly
                              (standard per-block FP4 quantization like V, no s_P1).
                              If False (default), use two-level quantization:
                              s_P1 = rowmax(P̃)/(448×6), then s_P2, P̂_2 = φ(P̃/s_P1).
        sm_scale: Softmax scale to pass through to the CUDA kernel. If None,
                  defaults to the kernel's 1/sqrt(D) scale.
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Output tensor [B, H, L, D]
    """
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal=is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    per_block_mean = delta_s.shape[2] > 1
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    o_fp4 = blockscaled_fp4_attn(qlist_from_cuda, klist_from_cuda, vlist_from_cuda, delta_s, KL, is_causal,
                                 per_block_mean, is_bf16, single_level_p_quant, sm_scale)[0][:, :, :QL, :].contiguous()
    return o_fp4
