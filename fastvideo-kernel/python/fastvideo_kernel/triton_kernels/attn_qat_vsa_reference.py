# SPDX-License-Identifier: Apache-2.0
"""Slow PyTorch reference for block-sparse FP4 attention as ``attn_qat_infer`` computes it.

This is the numeric specification that ``attn_qat_vsa_train`` (the Triton
fake-quant training kernel) must reproduce. It follows the deployed sm_120/121
kernel (``sageattn_blackwell_sparse_bshd`` on lists from
``vsa_tile_mask_to_fp4_blocks``) step by step:

* Q and K: NVFP4 E2M1 with one E4M3 scale per 16 channels along D, no global
  scale (``scaled_fp4_quant(x, ..., 1)``): ``s = e4m3(amax / 6)``, value
  ``e2m1(x / s) * s``, round-to-nearest-even, saturating.
* V: the same, but grouped along tokens (``scale_and_quant_fp4_transpose``).
* Each 128-row query block visits its listed 128-key blocks in ascending block
  order (the lists are stored descending and read from the last entry).
* P: the online softmax of ``softmax_fused.h``. For each visited block the row
  max is updated first, ``P~ = exp2(S * log2e * scale - m_scaled)``; per 16 keys
  ``s_P = max(P~) / 6`` is computed in fp32, ``P~ / s_P`` is rounded to E2M1 and
  the MMA multiplies by ``e4m3(s_P)``. Single-level P has ``m_scaled = m``;
  two-level P adds ``log2(1 / (448 * 6))`` so ``P~`` spans ``[0, 2688]``.
* The row sum uses the unquantized ``P~``; the output is ``acc / l``.
* Masked keys (``kv_valid`` halves) score ``-inf``; a fully masked 16-key group
  has ``s_P = 0`` and contributes exactly zero.

``first_block_max_floor`` models an uninitialized-register hazard in the
deployed kernel: on the first visited block, ``AbsMaxP`` is read before it is
written (``fmaxf(AbsMaxP, s)``). ``None`` assumes the compiler folds that to
the true max; a number (e.g. ``0.0``) floors the first block's group maxima
and hence the row max. The cross-device fidelity check decides which applies.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

QUANT_GROUP = 16
BLOCK = 128
HALF = BLOCK // 2
E4M3_MAX = 448.0
LOG2E = 1.4426950408889634
LOG2_INV_FP4_MAX = -2.584962500721156  # log2(1 / 6)
LOG2_INV_TWO_LEVEL = -11.392317422778762  # log2(1 / (448 * 6))


def e4m3_rn(x: torch.Tensor) -> torch.Tensor:
    """Round non-negative fp32 scales to E4M3 (RNE, saturating at 448) and back to fp32."""
    return x.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def e2m1_rn(y: torch.Tensor) -> torch.Tensor:
    """Round fp32 values to the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} (RNE, saturating)."""
    a = y.abs()
    q = torch.where(
        a <= 0.25, 0.0,
        torch.where(
            a < 0.75, 0.5,
            torch.where(
                a <= 1.25, 1.0,
                torch.where(a < 1.75, 1.5,
                            torch.where(a <= 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0,
                                                                                               6.0)))))))
    return torch.copysign(q, y)


def nvfp4_fake_quant_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Fake-quantize along the last dim in groups of 16 (no global scale); returns fp32."""
    if x.shape[-1] % QUANT_GROUP:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of {QUANT_GROUP}")
    xf = x.float().unflatten(-1, (-1, QUANT_GROUP))
    scale = e4m3_rn(xf.abs().amax(dim=-1, keepdim=True) / 6.0)
    inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    return (e2m1_rn(xf * inv) * scale).flatten(-2)


def fake_quant_qkv(q: torch.Tensor, k: torch.Tensor,
                   v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[B, H, L, D]`` inputs -> dequantized Q (along D), K (along D), V (along L); fp32."""
    v_hat = nvfp4_fake_quant_lastdim(v.transpose(-1, -2)).transpose(-1, -2)
    return nvfp4_fake_quant_lastdim(q), nvfp4_fake_quant_lastdim(k), v_hat


def smooth_keys(k: torch.Tensor) -> torch.Tensor:
    """K smoothing as ``preprocess_qkv`` (``k -= k.mean(dim=-2)``): per-head token mean, in the input dtype.

    The dtype matters: FP4 rounding is so input-sensitive that smoothing in
    fp32 instead of bf16 alone moves the attention output by ~8% rel-L2. The
    mean runs over every row given (tile-ordered, pad rows included); a
    deployed K-smoothing path must use the same rows.
    """
    return k - k.mean(dim=-2, keepdim=True)


def key_valid_mask(kv_valid: torch.Tensor | None, n_blocks: int, device: torch.device) -> torch.Tensor:
    """``[n_blocks, 128]`` bool: which key columns of each block are real tokens."""
    if kv_valid is None:
        return torch.ones(n_blocks, BLOCK, dtype=torch.bool, device=device)
    col = torch.arange(BLOCK, device=device)
    limit = torch.where(col < HALF, kv_valid[0::2, None], kv_valid[1::2, None])
    return (col % HALF)[None, :] < limit


def tile_mask_to_fp4_blocks(tile_mask: torch.Tensor,
                           tile_tokens: int,
                           tile_valid: torch.Tensor | None = None) -> tuple[
                               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pure-PyTorch ``attn_qat_infer.api.vsa_tile_mask_to_fp4_blocks`` for tile 64 or 128.

    The inference API imports sm_12x extensions at module load, so data center
    GPUs cannot call it.  The kernel works on 128x128 blocks.  With 64-token VSA
    tiles two query tiles and two key tiles are paired into one block; a block
    is listed when any of its four 64x64 quadrants is selected and ``q2k_quad``
    carries the selected quadrants (bit ``2 * query_half + key_half``).  With
    128-token tiles the mapping is 1:1 and no quadrant mask is needed.

    Returns ``(q2k_idx [B, H, M, N], q2k_num [B, H, M], kv_valid [2N],
    q2k_quad [B, H, M, N] | None)``; queries cover ``M * 128`` padded rows.
    """
    if tile_tokens % HALF or (tile_tokens > HALF and tile_tokens % BLOCK):
        raise ValueError(f"tile_tokens={tile_tokens} must be {HALF} or a multiple of {BLOCK}")
    batch, heads, n_tiles, _ = tile_mask.shape
    device = tile_mask.device
    halves_per_tile = tile_tokens // HALF
    half_mask = tile_mask
    if halves_per_tile > 1:
        half_mask = half_mask.repeat_interleave(halves_per_tile, dim=2).repeat_interleave(halves_per_tile, dim=3)
    n_halves = n_tiles * halves_per_tile
    if tile_valid is None:
        half_valid = torch.full((n_halves, ), HALF, device=device, dtype=torch.int32)
    else:
        offsets = torch.arange(halves_per_tile, device=device, dtype=torch.int32) * HALF
        half_valid = (tile_valid.to(torch.int32)[:, None] - offsets[None, :]).clamp(0, HALF).reshape(-1)
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
    q2k_quad = None
    if halves_per_tile == 1:
        q2k_quad = torch.zeros((batch, heads, n_blocks, n_blocks + 1), device=device, dtype=torch.uint8)
        q2k_quad.scatter_(-1, slot, quad.flip(-1))
        q2k_quad = q2k_quad[..., :n_blocks].contiguous()
    return q2k_idx, q2k_num, half_valid.contiguous(), q2k_quad


def tile128_mask_to_fp4_blocks(tile_mask: torch.Tensor,
                               tile_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible 128-token-tile wrapper."""
    idx, num, valid, quad = tile_mask_to_fp4_blocks(tile_mask, BLOCK, tile_valid)
    if quad is not None:
        raise AssertionError("128-token tiles must not produce a quadrant mask")
    return idx, num, valid


def invert_block_lists(q2k_idx: torch.Tensor, q2k_num: torch.Tensor,
                       n_kblocks: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query->key lists to key->query lists, keeping each entry's slot in its query list.

    Returns ``(k2q_idx [B, H, NK, M], k2q_slot [B, H, NK, M], k2q_num [B, H, NK])`` (int32).
    """
    batch, heads, n_qblocks, width = q2k_idx.shape
    device = q2k_idx.device
    slots = torch.arange(width, device=device, dtype=torch.int32).expand_as(q2k_idx)
    listed = slots < q2k_num[..., None]
    slot_of = torch.full((batch, heads, n_qblocks, n_kblocks + 1), -1, device=device, dtype=torch.int32)
    slot_of.scatter_(-1, torch.where(listed, q2k_idx.long(), n_kblocks), torch.where(listed, slots, -1))
    by_key = slot_of[..., :n_kblocks].transpose(-1, -2)  # [B, H, NK, M]
    present = by_key >= 0
    k2q_num = present.sum(-1, dtype=torch.int32)
    position = torch.where(present, present.cumsum(-1) - 1, n_qblocks).long()
    queries = torch.arange(n_qblocks, device=device, dtype=torch.int32).expand_as(by_key)
    k2q_idx = torch.zeros((batch, heads, n_kblocks, n_qblocks + 1), device=device, dtype=torch.int32)
    k2q_slot = torch.zeros_like(k2q_idx)
    k2q_idx.scatter_(-1, position, queries)
    k2q_slot.scatter_(-1, position, by_key)
    return k2q_idx[..., :n_qblocks].contiguous(), k2q_slot[..., :n_qblocks].contiguous(), k2q_num.contiguous()


def visit_order(q2k_idx: torch.Tensor, q2k_num: torch.Tensor, b: int, h: int, m: int) -> list[int]:
    """Blocks query block ``m`` visits, in kernel order (last list entry first)."""
    count = int(q2k_num[b, h, m])
    return [int(block) for block in q2k_idx[b, h, m, :count].flip(0)]


@torch.no_grad()
def fp4_vsa_attention_reference(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                q2k_idx: torch.Tensor,
                                q2k_num: torch.Tensor,
                                kv_valid: torch.Tensor | None = None,
                                q2k_quad: torch.Tensor | None = None,
                                *,
                                two_level_p: bool = False,
                                smooth_k: bool = False,
                                quantize: bool = True,
                                first_block_max_floor: float | None = None,
                                sm_scale: float | None = None,
                                return_dv_weights: bool = False):
    """Emulate the deployed block-sparse FP4 attention on ``[B, H, L, D]`` inputs (L a multiple of 128).

    ``quantize=False`` runs the same visit order with exact softmax numerics
    (BF16-input reference). Returns ``[B, H, L, D]`` fp32, and with
    ``return_dv_weights`` also the dense ``[B, H, L, L]`` matrix ``W`` with
    ``out = W @ v_hat`` (the forward-consistent dV weights).
    """
    batch, heads, seq_len, dim = q.shape
    if seq_len % BLOCK:
        raise ValueError(f"sequence length {seq_len} must be a multiple of {BLOCK}")
    scale_log2 = (sm_scale if sm_scale is not None else 1.0 / math.sqrt(dim)) * LOG2E
    if smooth_k:
        k = smooth_keys(k)
    if quantize:
        q_hat, k_hat, v_hat = fake_quant_qkv(q, k, v)
    else:
        q_hat, k_hat, v_hat = q.float(), k.float(), v.float()
    offset = LOG2_INV_TWO_LEVEL if two_level_p else 0.0
    n_blocks = seq_len // BLOCK
    valid = key_valid_mask(kv_valid, n_blocks, q.device)
    out = torch.zeros(batch, heads, seq_len, dim, dtype=torch.float32, device=q.device)
    weights = torch.zeros(batch, heads, seq_len, seq_len, device=q.device) if return_dv_weights else None
    for b in range(batch):
        for h in range(heads):
            for m in range(n_blocks):
                rows = slice(m * BLOCK, (m + 1) * BLOCK)
                row_max = torch.full((BLOCK, ), -math.inf, device=q.device)
                row_sum = torch.zeros(BLOCK, device=q.device)
                acc = torch.zeros(BLOCK, dim, device=q.device)
                visited: list[tuple[int, torch.Tensor, torch.Tensor]] = []
                count = int(q2k_num[b, h, m])
                entries = list(range(count))[::-1]
                for position, entry in enumerate(entries):
                    n = int(q2k_idx[b, h, m, entry])
                    cols = slice(n * BLOCK, (n + 1) * BLOCK)
                    scores = q_hat[b, h, rows] @ k_hat[b, h, cols].T
                    scores = scores.masked_fill(~valid[n][None, :], -math.inf)
                    if q2k_quad is not None:
                        quad = int(q2k_quad[b, h, m, entry])
                        half = torch.arange(BLOCK, device=q.device) // HALF
                        qvalid = ((quad >> (half[:, None] * 2 + half[None, :])) & 1).bool()
                        scores = scores.masked_fill(~qvalid, -math.inf)
                    group_max = scores.unflatten(-1, (-1, QUANT_GROUP)).amax(dim=-1)
                    if position == 0 and first_block_max_floor is not None:
                        group_max = group_max.clamp(min=first_block_max_floor)
                    new_max = torch.maximum(row_max, group_max.amax(dim=-1))
                    m_scaled = new_max * scale_log2 + offset
                    p_tilde = torch.exp2(scores * scale_log2 - m_scaled[:, None])
                    alpha = torch.exp2((row_max - new_max) * scale_log2)
                    alpha = torch.where(torch.isinf(row_max), torch.zeros_like(alpha), alpha)
                    row_sum = row_sum * alpha + p_tilde.sum(dim=-1)
                    if quantize:
                        s_p = torch.exp2(group_max * scale_log2 - m_scaled[:, None] + LOG2_INV_FP4_MAX)
                        s_full = s_p.repeat_interleave(QUANT_GROUP, dim=-1)
                        ratio = torch.where(s_full == 0, torch.zeros_like(p_tilde), p_tilde / s_full)
                        p_hat = e2m1_rn(ratio) * e4m3_rn(s_full)
                    else:
                        p_hat = p_tilde
                    acc = acc * alpha[:, None] + p_hat @ v_hat[b, h, cols]
                    visited.append((n, p_hat, new_max))
                    row_max = new_max
                inv = torch.where((row_sum == 0) | torch.isnan(row_sum), torch.zeros_like(row_sum), 1.0 / row_sum)
                out[b, h, rows] = acc * inv[:, None]
                if weights is not None:
                    for n, p_hat, block_max in visited:
                        decay = torch.exp2((block_max - row_max) * scale_log2) * inv
                        weights[b, h, rows, n * BLOCK:(n + 1) * BLOCK] = p_hat * decay[:, None]
    if weights is not None:
        return out, weights
    return out


def dense_block_mask(q2k_idx: torch.Tensor, q2k_num: torch.Tensor, seq_len: int,
                     kv_valid: torch.Tensor | None = None) -> torch.Tensor:
    """Token-level ``[B, H, L, L]`` bool mask of the listed blocks (and valid keys)."""
    batch, heads, n_blocks, width = q2k_idx.shape
    listed = torch.arange(width, device=q2k_idx.device) < q2k_num[..., None]
    block_mask = torch.zeros(batch, heads, n_blocks, n_blocks + 1, dtype=torch.bool, device=q2k_idx.device)
    block_mask.scatter_(-1, torch.where(listed, q2k_idx.long(), n_blocks), True)
    block_mask = block_mask[..., :n_blocks]
    token = block_mask.repeat_interleave(BLOCK, dim=-2).repeat_interleave(BLOCK, dim=-1)[..., :seq_len, :seq_len]
    valid = key_valid_mask(kv_valid, n_blocks, q2k_idx.device).reshape(-1)[:seq_len]
    return token & valid


def ste_gradients_reference(q_hat: torch.Tensor, k_hat: torch.Tensor, v_hat: torch.Tensor, token_mask: torch.Tensor,
                            dv_weights: torch.Tensor, grad_out: torch.Tensor,
                            sm_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The training kernel's STE gradients, computed densely in fp32.

    dQ and dK are the exact gradients of the high-precision surrogate
    ``softmax(q_hat k_hat^T) v_hat`` on the sparse mask; dV uses the forward's
    quantized, online-softmax weights (forward-consistent dV).
    """
    q_hat, k_hat, v_hat = (t.detach().float().requires_grad_(True) for t in (q_hat, k_hat, v_hat))
    scores = (q_hat @ k_hat.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores.masked_fill(~token_mask, -math.inf), dim=-1)
    surrogate = probs @ v_hat
    dq, dk = torch.autograd.grad(surrogate, (q_hat, k_hat), grad_out.float())
    dv = dv_weights.transpose(-1, -2) @ grad_out.float()
    return dq, dk, dv
