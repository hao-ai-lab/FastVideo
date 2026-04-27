# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the fused RoPE Triton kernel."""
from __future__ import annotations

import pytest
import torch

from fastvideo_kernel.triton_kernels.rope_triton import apply_rope


def _rotate_half_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos_e = cos.unsqueeze(-2)
    sin_e = sin.unsqueeze(-2)
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    return (x.float() * cos_e + x_rot * sin_e).type_as(x)


def _gptj_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos_e = cos.unsqueeze(-2)
    sin_e = sin.unsqueeze(-2)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = (x1.float() * cos_e - x2.float() * sin_e).type_as(x)
    o2 = (x2.float() * cos_e + x1.float() * sin_e).type_as(x)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _neox_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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

_REFS = {
    "rotate_half": (_rotate_half_ref, False, 1),  # (ref, is_neox, rope_dim_scale) -> cos dim = D*scale
    "gptj": (_gptj_ref, False, 0),
    "neox": (_neox_ref, True, 0),
}


def _cos_sin(T: int, D: int, mode: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    rope_dim = D if mode == "rotate_half" else D // 2
    cos = torch.randn(T, rope_dim, device="cuda", dtype=dtype)
    sin = torch.randn(T, rope_dim, device="cuda", dtype=dtype)
    return cos, sin


# Hand-picked cases collectively covering all 3 modes x 3 dtypes x edge shapes
# (T=1, non-pow2 D, odd T, large D, realistic DiT-scale).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "mode,dtype,T,H,D",
    [
        ("rotate_half", torch.float32, 1, 1, 64),        # minimal edges
        ("rotate_half", torch.float16, 17, 8, 96),       # non-pow2 D + odd T
        ("rotate_half", torch.bfloat16, 2048, 24, 128),  # realistic DiT scale
        ("rotate_half", torch.bfloat16, 64, 4, 256),     # large head_size
        ("gptj", torch.float32, 17, 4, 128),
        ("gptj", torch.float16, 1, 2, 64),
        ("gptj", torch.bfloat16, 513, 8, 128),
        ("neox", torch.float32, 17, 4, 128),
        ("neox", torch.float16, 1, 2, 64),
        ("neox", torch.bfloat16, 513, 8, 128),
    ],
)
def test_rope_fused(mode, dtype, T, H, D):
    torch.manual_seed(hash((mode, T, H, D)) & 0xFFFF)
    x = torch.randn(T, H, D, device="cuda", dtype=dtype)
    cos, sin = _cos_sin(T, D, mode, dtype)

    ref_fn, is_neox, _ = _REFS[mode]
    out = apply_rope(x, cos, sin, is_neox_style=is_neox)
    assert out is not None, f"eligible case unexpectedly rejected: {mode}, {dtype}, {(T, H, D)}"
    ref = ref_fn(x, cos, sin)
    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_ineligible_cpu_tensor():
    x = torch.randn(4, 2, 64)
    cos = torch.randn(4, 64)
    sin = torch.randn(4, 64)
    assert apply_rope(x, cos, sin, is_neox_style=False) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "label,x_shape,cos_shape",
    [
        ("mismatched_rope_dim", (4, 2, 64), (4, 48)),
        ("odd_head_size",       (4, 2, 63), (4, 63)),
        ("mismatched_tokens",   (4, 2, 64), (8, 64)),
    ],
)
def test_ineligible_shape(label, x_shape, cos_shape):
    x = torch.randn(*x_shape, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(*cos_shape, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(*cos_shape, device="cuda", dtype=torch.bfloat16)
    assert apply_rope(x, cos, sin, is_neox_style=False) is None
