# SPDX-License-Identifier: Apache-2.0
"""FASTVIDEO_H3_VAE_FUSED_QK_NORM_ROPE: fused VAE-decoder QK RMSNorm + partial RoPE."""

import pytest
import torch

from fastvideo.models.vaes.minimax_h3_qk_norm_rope import qk_norm_rope_
from fastvideo.models.vaes.minimax_h3_video import MiniMaxH3VideoAttention


def _eager_prologue(query, key, cos, sin, eps):
    """The VAE decoder's eager QK prologue (the fused kernel's reference)."""
    def rms_norm(x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)

    query = rms_norm(query).to(query.dtype)
    key = rms_norm(key).to(key.dtype)
    cos = cos.to(query.dtype)
    sin = sin.to(query.dtype)
    rotary_dim = cos.shape[-1]

    def rotate(x):
        rotary, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat([-second, first], dim=-1)
        return torch.cat([rotary * cos + rotated * sin, passthrough], dim=-1)

    return rotate(query), rotate(key)


def test_fused_prologue_is_a_cpu_no_op(monkeypatch) -> None:
    """Without CUDA the env flag must leave the eager path (and its output) untouched."""
    torch.manual_seed(20260819)
    attention = MiniMaxH3VideoAttention(dim=32, heads=2, dim_head=16, bias=True).eval()
    hidden_states = torch.randn(2, 7, 32)
    rotary_emb = (torch.randn(2, 7, 1, 12), torch.randn(2, 7, 1, 12))

    with torch.no_grad():
        monkeypatch.delenv("FASTVIDEO_H3_VAE_FUSED_QK_NORM_ROPE", raising=False)
        eager = attention(hidden_states, rotary_emb)
        monkeypatch.setenv("FASTVIDEO_H3_VAE_FUSED_QK_NORM_ROPE", "1")
        flagged = attention(hidden_states, rotary_emb)

    assert torch.equal(eager, flagged)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_prologue_matches_the_eager_arithmetic(dtype: torch.dtype) -> None:
    """Production geometry: 32 heads x 64 dims, partial RoPE on 48 dims."""
    torch.manual_seed(20260820)
    batch, seq, heads, head_dim, rope_dim = 8, 337, 32, 64, 48
    query = (torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype) * 0.02)
    key = (torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype) * 0.02)
    cos = torch.rand(seq, rope_dim, device="cuda", dtype=torch.float32)
    sin = torch.rand(seq, rope_dim, device="cuda", dtype=torch.float32)
    reference_query, reference_key = _eager_prologue(query, key, cos[None, :, None], sin[None, :, None], 1e-5)

    fused_query, fused_key = query.clone(), key.clone()
    qk_norm_rope_(fused_query, fused_key, cos[None, :, None].expand(batch, -1, -1, -1).contiguous(),
                  sin[None, :, None].expand(batch, -1, -1, -1).contiguous(), heads, head_dim, rope_dim)

    for name, fused, reference in (("q", fused_query, reference_query), ("k", fused_key, reference_key)):
        difference = (fused.float() - reference.float()).abs()
        bit_equal = (fused == reference).float().mean().item()
        relative_l2 = (difference.norm() / reference.float().norm()).item()
        # Only the fp32 RMSNorm reduction order differs; RoPE rounds exactly like eager.
        assert bit_equal > 0.9995, name
        assert relative_l2 < 1e-5, name
        assert difference.max().item() <= 4 * torch.finfo(dtype).eps * reference.float().abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel requires CUDA")
@torch.inference_mode()
def test_attention_forward_matches_eager_within_one_ulp(monkeypatch) -> None:
    """A 32-head attention block produces the same output with the fused prologue.

    The prologue's RMSNorm reduction order is the only difference, so the
    attention output lands within one activation-dtype ULP of the eager path.
    """
    torch.manual_seed(20260821)
    attention = MiniMaxH3VideoAttention(dim=2048, heads=32, dim_head=64, bias=True).cuda().eval()
    hidden_states = torch.randn(1, 129, 2048, device="cuda")
    rotary_emb = (torch.randn(1, 129, 1, 48, device="cuda"), torch.randn(1, 129, 1, 48, device="cuda"))
    with torch.autocast("cuda", dtype=torch.float16):
        monkeypatch.delenv("FASTVIDEO_H3_VAE_FUSED_QK_NORM_ROPE", raising=False)
        eager = attention(hidden_states, rotary_emb)
        monkeypatch.setenv("FASTVIDEO_H3_VAE_FUSED_QK_NORM_ROPE", "1")
        fused = attention(hidden_states, rotary_emb)

    difference = (fused.float() - eager.float()).abs().max().item()
    assert difference <= 2 * torch.finfo(torch.float16).eps * eager.float().abs().max().item()
