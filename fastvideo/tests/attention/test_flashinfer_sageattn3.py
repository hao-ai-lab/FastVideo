# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the FLASHINFER_SAGE_ATTN3 backend.

flashinfer is deliberately NOT installed in this environment: these tests
exercise the registration wiring and the torch-SDPA fallback path, which is
exactly what runs on machines without the SM120 kernel (e.g. shape collection
on DGX Spark). No GPU required.
"""

from __future__ import annotations

import math

import pytest
import torch

from fastvideo.attention.backends import flashinfer_sageattn3 as flashinfer_backend
from fastvideo.attention.backends.flashinfer_sageattn3 import (FlashInferSageAttention3Backend,
                                                               FlashInferSageAttention3Impl)
from fastvideo.attention.selector import backend_name_to_enum
from fastvideo.platforms.cuda import CudaPlatform
from fastvideo.platforms.interface import AttentionBackendEnum


def test_enum_and_name_roundtrip():
    enum_val = backend_name_to_enum("FLASHINFER_SAGE_ATTN3")
    assert enum_val is AttentionBackendEnum.FLASHINFER_SAGE_ATTN3
    assert FlashInferSageAttention3Backend.get_name() == "FLASHINFER_SAGE_ATTN3"
    assert FlashInferSageAttention3Backend.get_impl_cls() is FlashInferSageAttention3Impl
    assert FlashInferSageAttention3Backend.get_supported_head_sizes() == [64, 128]


def test_cuda_platform_pin_returns_backend_cls():
    cls_path = CudaPlatform.get_attn_backend_cls(AttentionBackendEnum.FLASHINFER_SAGE_ATTN3,
                                                 head_size=128,
                                                 dtype=torch.bfloat16)
    assert cls_path == ("fastvideo.attention.backends.flashinfer_sageattn3.FlashInferSageAttention3Backend")


def _manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                      causal: bool) -> torch.Tensor:
    """Reference attention on [B, H, L, D] tensors."""
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        seq_len = q.shape[2]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float())


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fallback_forward_matches_reference(causal: bool, head_dim: int):
    torch.manual_seed(0)
    batch, seq_len, num_heads = 2, 17, 4
    scale = 1.0 / math.sqrt(head_dim)
    impl = FlashInferSageAttention3Impl(num_heads=num_heads,
                                        head_size=head_dim,
                                        causal=causal,
                                        softmax_scale=scale)

    # FastVideo impl convention: [B, L, H, D]
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = impl.forward(q, k, v, attn_metadata=None)
    assert out.shape == q.shape

    ref = _manual_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale,
                            causal).transpose(1, 2)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_fallback_forward_gqa():
    torch.manual_seed(1)
    batch, seq_len, head_dim = 1, 9, 64
    scale = 1.0 / math.sqrt(head_dim)
    impl = FlashInferSageAttention3Impl(num_heads=4, head_size=head_dim, causal=False, softmax_scale=scale)

    q = torch.randn(batch, seq_len, 4, head_dim)
    k = torch.randn(batch, seq_len, 2, head_dim)
    v = torch.randn_like(k)

    out = impl.forward(q, k, v, attn_metadata=None)

    k_rep = k.repeat_interleave(2, dim=2)
    v_rep = v.repeat_interleave(2, dim=2)
    ref = _manual_attention(q.transpose(1, 2), k_rep.transpose(1, 2), v_rep.transpose(1, 2), scale,
                            causal=False).transpose(1, 2)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_flashinfer_quantize_disables_ambient_autocast(monkeypatch):
    """Keep FlashInfer's qk_correction fp32 inside the bf16 pipeline context."""
    correction_dtype = None

    def fake_quantize(q, k, v):
        nonlocal correction_dtype
        correction = torch.matmul(q.float(), k.float().transpose(-2, -1))
        correction_dtype = correction.dtype
        return q, k, v, q, k, v, correction

    def fake_forward(q, _k, _v, _qs, _ks, _vs, correction, **_kwargs):
        assert correction.dtype == torch.float32
        return q, None

    monkeypatch.setattr(flashinfer_backend, "_can_use_flashinfer", lambda *_args: True)
    monkeypatch.setattr(flashinfer_backend, "nvfp4_attention_sm120_quantize_qkv", fake_quantize)
    monkeypatch.setattr(flashinfer_backend, "nvfp4_attention_sm120_fwd", fake_forward)

    impl = FlashInferSageAttention3Impl(num_heads=2,
                                        head_size=64,
                                        causal=False,
                                        softmax_scale=1 / math.sqrt(64))
    q = torch.randn(1, 8, 2, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = impl.forward(q, q, q, attn_metadata=None)

    assert correction_dtype == torch.float32
    assert out.shape == q.shape
