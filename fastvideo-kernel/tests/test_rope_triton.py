# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the fused RoPE Triton kernel."""
from __future__ import annotations

import pytest
import torch

from fastvideo_kernel.triton_kernels.rope_triton import apply_rope


def _rotate_half_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference for rotate_half style (rope_dim == head_size)."""
    cos_e = cos.unsqueeze(-2)
    sin_e = sin.unsqueeze(-2)
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    return (x.float() * cos_e + x_rot * sin_e).type_as(x)


def _gptj_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference for GPT-J interleaved style (rope_dim == head_size // 2)."""
    cos_e = cos.unsqueeze(-2)
    sin_e = sin.unsqueeze(-2)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = (x1.float() * cos_e - x2.float() * sin_e).type_as(x)
    o2 = (x2.float() * cos_e + x1.float() * sin_e).type_as(x)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _neox_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference for Neox split-half style (rope_dim == head_size // 2)."""
    cos_e = cos.unsqueeze(-2)
    sin_e = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = (x1.float() * cos_e - x2.float() * sin_e).type_as(x)
    o2 = (x2.float() * cos_e + x1.float() * sin_e).type_as(x)
    return torch.cat((o1, o2), dim=-1)


_TOL = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 17, 2048])
@pytest.mark.parametrize("num_heads", [1, 8, 24])
@pytest.mark.parametrize("head_size", [64, 96, 128, 256])
def test_rotate_half(dtype, num_tokens, num_heads, head_size):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=dtype)
    cos = torch.randn(num_tokens, head_size, device="cuda", dtype=dtype)
    sin = torch.randn(num_tokens, head_size, device="cuda", dtype=dtype)

    out = apply_rope(x, cos, sin, is_neox_style=False)
    assert out is not None
    ref = _rotate_half_ref(x, cos, sin)
    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 513])
@pytest.mark.parametrize("head_size", [64, 128])
def test_gptj(dtype, num_tokens, head_size):
    torch.manual_seed(1)
    num_heads = 4
    x = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=dtype)
    cos = torch.randn(num_tokens, head_size // 2, device="cuda", dtype=dtype)
    sin = torch.randn(num_tokens, head_size // 2, device="cuda", dtype=dtype)

    out = apply_rope(x, cos, sin, is_neox_style=False)
    assert out is not None
    ref = _gptj_ref(x, cos, sin)
    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 513])
@pytest.mark.parametrize("head_size", [64, 128])
def test_neox(dtype, num_tokens, head_size):
    torch.manual_seed(2)
    num_heads = 4
    x = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=dtype)
    cos = torch.randn(num_tokens, head_size // 2, device="cuda", dtype=dtype)
    sin = torch.randn(num_tokens, head_size // 2, device="cuda", dtype=dtype)

    out = apply_rope(x, cos, sin, is_neox_style=True)
    assert out is not None
    ref = _neox_ref(x, cos, sin)
    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_ineligible_returns_none_cpu():
    x = torch.randn(4, 2, 64)
    cos = torch.randn(4, 64)
    sin = torch.randn(4, 64)
    assert apply_rope(x, cos, sin, is_neox_style=False) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ineligible_mismatched_rope_dim():
    x = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(4, 48, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(4, 48, device="cuda", dtype=torch.bfloat16)
    assert apply_rope(x, cos, sin, is_neox_style=False) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ineligible_odd_head_size():
    x = torch.randn(4, 2, 63, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(4, 63, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(4, 63, device="cuda", dtype=torch.bfloat16)
    assert apply_rope(x, cos, sin, is_neox_style=False) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_dtype_cos_sin():
    """cos/sin in fp32, x in bf16 is the common DiT case after freqs_cis construction."""
    torch.manual_seed(3)
    x = torch.randn(128, 8, 128, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

    out = apply_rope(x, cos, sin, is_neox_style=False)
    assert out is not None
    ref = _rotate_half_ref(x, cos, sin)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
