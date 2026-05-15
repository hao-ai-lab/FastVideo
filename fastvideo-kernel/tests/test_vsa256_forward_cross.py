"""Three-way parity for the VSA-256 forward: torch reference vs CuTe vs Triton."""

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
    q_block = int(q_var[0].item())
    kv_block = int(kv_var[0].item())

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

    topk_idx = torch.topk(scores, topk_logical, dim=-1).indices
    block_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)
    token_mask = (
        block_mask
        .repeat_interleave(q_block, dim=2)
        .repeat_interleave(kv_block, dim=3)
        .to(torch.bool)
    )

    qf, kf, vf = q.float(), k.float(), v.float()
    logits = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(dim)
    logits = logits.masked_fill(~token_mask, float("-inf"))
    prob = torch.softmax(logits, dim=-1)
    out_s = torch.matmul(prob, vf).to(q.dtype)
    return out_c + out_s


def _metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a - b).abs()
    avg_abs = diff.mean().item()
    max_rel = (diff.max() / (a.abs().mean() + 1e-6)).item()
    return avg_abs, max_rel


@pytest.mark.cuda
def test_vsa256_forward_cross_torch_cute_triton(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    bsz, heads, dim = 1, 8, 128
    q_blocks_256, kv_blocks_256 = 8, 12
    topk_logical = 2
    q_block = 256
    kv_block = 256

    sq = q_blocks_256 * q_block
    skv = kv_blocks_256 * kv_block
    q = torch.randn(bsz, heads, sq, dim, device=device, dtype=dtype)
    k = torch.randn(bsz, heads, skv, dim, device=device, dtype=dtype)
    v = torch.randn(bsz, heads, skv, dim, device=device, dtype=dtype)
    q_var = torch.full((q_blocks_256,), q_block, dtype=torch.int32, device=device)
    kv_var = torch.full((kv_blocks_256,), kv_block, dtype=torch.int32, device=device)

    out_torch = _torch_vsa256_reference(q, k, v, q_var, kv_var, topk_logical)

    # CuTe (default).
    out_cute = video_sparse_attn(
        q, k, v,
        kv_var, q_var, topk_logical,
        block_size=(4, 8, 8),
        compress_attn_weight=None,
    )

    # Triton via route-A 256->64 expansion.
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
    out_triton = video_sparse_attn(
        q, k, v,
        kv_var, q_var, topk_logical,
        block_size=(4, 8, 8),
        compress_attn_weight=None,
    )

    for t in (out_torch, out_cute, out_triton):
        assert torch.isfinite(t).all().item()

    torch_vs_cute = _metrics(out_torch, out_cute)
    torch_vs_triton = _metrics(out_torch, out_triton)
    cute_vs_triton = _metrics(out_cute, out_triton)
    print(
        "[cross-forward] "
        f"torch_vs_cute(avg_abs={torch_vs_cute[0]:.6e}, max_rel={torch_vs_cute[1]:.6e}), "
        f"torch_vs_triton(avg_abs={torch_vs_triton[0]:.6e}, max_rel={torch_vs_triton[1]:.6e}), "
        f"cute_vs_triton(avg_abs={cute_vs_triton[0]:.6e}, max_rel={cute_vs_triton[1]:.6e})"
    )

    assert torch_vs_cute[0] < 1e-3 and torch_vs_cute[1] < 0.2
    assert torch_vs_triton[0] < 1e-3 and torch_vs_triton[1] < 0.2
    assert cute_vs_triton[0] < 1e-3 and cute_vs_triton[1] < 0.2
