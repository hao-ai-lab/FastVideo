"""VSA-256 CuTe correctness with variable KV block sizes (<256)."""

from __future__ import annotations

import math

import pytest
import torch

from fastvideo_kernel import video_sparse_attn


def _torch_vsa256_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_var: torch.Tensor,
    kv_var: torch.Tensor,
    topk_logical: int,
) -> torch.Tensor:
    bsz, heads, _sq, dim = q.shape
    q_blocks = q_var.numel()
    kv_blocks = kv_var.numel()
    q_block = q.shape[2] // q_blocks
    kv_block = k.shape[2] // kv_blocks

    q_c = q.view(bsz, heads, q_blocks, q_block, dim)
    k_c = k.view(bsz, heads, kv_blocks, kv_block, dim)
    v_c = v.view(bsz, heads, kv_blocks, kv_block, dim)
    q_c = (q_c.float().sum(dim=3) / q_var.view(1, 1, -1, 1)).to(q.dtype)
    k_c = (k_c.float().sum(dim=3) / kv_var.view(1, 1, -1, 1)).to(k.dtype)
    v_c = (v_c.float().sum(dim=3) / kv_var.view(1, 1, -1, 1)).to(v.dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / math.sqrt(dim)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)
    out_c = (
        out_c.view(bsz, heads, q_blocks, 1, dim)
        .repeat(1, 1, 1, q_block, 1)
        .view_as(q)
    )

    with torch.no_grad():
        topk_idx = torch.topk(scores.detach(), topk_logical, dim=-1).indices
        block_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)
        block_token_idx = torch.arange(kv_block, device=kv_var.device, dtype=torch.int32)
        kv_token_valid_by_block = (
            block_token_idx.view(1, -1) < kv_var.to(torch.int32).view(-1, 1)
        ).to(torch.bool)
        kv_token_valid = kv_token_valid_by_block.reshape(1, 1, 1, kv_blocks * kv_block)
        token_mask = (
            block_mask
            .repeat_interleave(q_block, dim=2)
            .repeat_interleave(kv_block, dim=3)
        )
        token_mask = token_mask & kv_token_valid

    qf, kf, vf = q.float(), k.float(), v.float()
    logits = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(dim)
    logits = logits.masked_fill(~token_mask, float("-inf"))
    prob = torch.softmax(logits, dim=-1)
    out_s = torch.matmul(prob, vf).to(q.dtype)
    return out_c + out_s


@pytest.mark.cuda
def test_vsa256_cute_variable_block_size_vs_torch_ref() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    bsz, heads, dim = 1, 8, 128
    q_blocks_256, kv_blocks_256 = 8, 12
    q_block = 256
    kv_block = 256
    topk_logical = 2
    sq = q_blocks_256 * q_block
    skv = kv_blocks_256 * kv_block

    q = torch.randn(bsz, heads, sq, dim, device=device, dtype=dtype)
    k = torch.randn(bsz, heads, skv, dim, device=device, dtype=dtype)
    v = torch.randn(bsz, heads, skv, dim, device=device, dtype=dtype)

    q_var = torch.full((q_blocks_256,), q_block, dtype=torch.int32, device=device)
    kv_var = torch.randint(16, kv_block + 1, (kv_blocks_256,), dtype=torch.int32, device=device)
    token_idx = torch.arange(kv_block, device=device, dtype=torch.int32)
    kv_valid = token_idx.view(1, -1) < kv_var.view(-1, 1)
    kv_valid = kv_valid.view(1, 1, kv_blocks_256, kv_block, 1)
    kv_valid = kv_valid.expand(bsz, heads, kv_blocks_256, kv_block, dim).reshape(bsz, heads, skv, dim)
    k = k * kv_valid.to(k.dtype)
    v = v * kv_valid.to(v.dtype)

    out = video_sparse_attn(
        q, k, v,
        kv_var,
        q_var,
        topk_logical,
        block_size=(4, 8, 8),
        compress_attn_weight=None,
    )
    out_ref = _torch_vsa256_reference(q, k, v, q_var, kv_var, topk_logical)

    diff = (out - out_ref).abs()
    avg_abs = diff.mean().item()
    max_rel = (diff.max() / (out_ref.abs().mean() + 1e-6)).item()
    print(
        f"[vsa256-cute-vbs] kv_var[min={int(kv_var.min().item())}, "
        f"max={int(kv_var.max().item())}], "
        f"avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}"
    )

    assert avg_abs < 1e-3 and max_rel < 0.2
