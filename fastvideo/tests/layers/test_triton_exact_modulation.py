# SPDX-License-Identifier: Apache-2.0
"""The exact Wan kernels must equal the eager chains they replace, bit for bit.

Unlike test_triton_fused_norm.py, which allows one bf16 ulp for the fused-norm
kernel, every GPU comparison here is ``torch.equal`` on the raw bit patterns.
Each test writes the eager expression the way fastvideo/models/wan/transformer.py
and fastvideo/layers/layernorm.py write it.
"""
import pytest
import torch
import torch.nn.functional as F

from fastvideo.layers import triton_exact_modulation as exact
from fastvideo.layers.layernorm import FP32LayerNorm

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
EPS = 1e-6


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _stream(batch: int, seq: int, hidden: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    residual = (torch.randn(batch, seq, hidden, device="cuda") * 3).to(torch.bfloat16)
    x = torch.randn(batch, seq, hidden, device="cuda").to(torch.bfloat16)
    # Edge values: signed zeros, subnormal-scale, large, and small magnitudes.
    residual[0, 0, :8] = torch.tensor([0., -0., 1e-30, -1e-30, 200., -200., 1e-3, -1e-3],
                                      device="cuda").to(torch.bfloat16)
    return residual, x


def _modulation(batch: int, seq: int, hidden: int, per_token: bool, seed: int):
    """Six fp32 chunks of one scale_shift_table + temb sum, as strided views like the Wan block makes them."""
    torch.manual_seed(seed + 100)
    rows = seq if per_token else 1
    table = torch.randn(batch, rows, 6, hidden, device="cuda")
    return tuple(t.squeeze(2) for t in table.chunk(6, dim=2))


SHAPES = [
    pytest.param(1, 9360, 1536, False, id="wan2.1-1.3b-sp2"),
    pytest.param(1, 9360, 1536, True, id="wan2.2-ti2v-per-token"),
    pytest.param(2, 777, 1536, False, id="batch2-odd-seq"),
    pytest.param(1, 1024, 3072, True, id="wan2.2-5b"),
    pytest.param(1, 1024, 5120, False, id="wan-14b"),
    pytest.param(1, 300, 1000, False, id="non-power-of-2-hidden"),
]


@cuda_only
@pytest.mark.parametrize("batch,seq,hidden,per_token", SHAPES)
def test_site1_norm1_modulate(batch, seq, hidden, per_token):
    """Wan pre-self-attn: ``(norm1(h.float()) * (1 + scale) + shift).to(orig_dtype)``."""
    residual, _ = _stream(batch, seq, hidden, seed=1)
    shift, scale = _modulation(batch, seq, hidden, per_token, seed=1)[:2]
    norm1 = FP32LayerNorm(hidden, EPS, elementwise_affine=False).cuda()
    with torch.inference_mode():
        eager = (norm1(residual.float()) * (1 + scale) + shift).to(torch.bfloat16)
        normed = norm1(residual.float())
        assert exact.supports_modulate(normed, scale, shift)
        assert _same_bits(eager, exact.modulate(normed, scale, shift, round_input=False))
        assert _same_bits(eager, exact.modulate_to(normed, scale, shift, torch.bfloat16))


@cuda_only
@pytest.mark.parametrize("batch,seq,hidden,per_token", SHAPES)
def test_site2_gated_residual_affine_norm(batch, seq, hidden, per_token):
    """Wan post-self-attn: fp32 gate, affine FP32LayerNorm, no modulation, caller casts both outputs."""
    residual, x = _stream(batch, seq, hidden, seed=2)
    gate = _modulation(batch, seq, hidden, per_token, seed=2)[2]
    norm = FP32LayerNorm(hidden, EPS, elementwise_affine=True).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden) * 0.02 + 1.0)
        norm.bias.copy_(torch.randn(hidden) * 0.01)
    with torch.inference_mode():
        eager_sum = residual + x * gate
        eager_normed, eager_res = norm(eager_sum).to(torch.bfloat16), eager_sum.to(torch.bfloat16)
        assert exact.supports_residual_norm(residual, x, gate, None, None)
        res16, sum32 = exact.gate_residual(residual, x, gate, fp32_copy=True)
        assert _same_bits(eager_sum, sum32)
        assert _same_bits(eager_res, res16)
        out, res = exact.residual_norm_modulate(residual, x, gate, None, None, norm)
        assert out.dtype == torch.float32  # eager promotion dtype; the caller casts
        assert _same_bits(eager_normed, out.to(torch.bfloat16))
        assert _same_bits(eager_res, res)


@cuda_only
@pytest.mark.parametrize("batch,seq,hidden,per_token", SHAPES)
def test_site3_residual_norm_modulate(batch, seq, hidden, per_token):
    """Wan post-cross-attn: gate 1, no-affine FP32LayerNorm, fp32 modulation, caller casts the output."""
    residual, x = _stream(batch, seq, hidden, seed=3)
    shift, scale = _modulation(batch, seq, hidden, per_token, seed=3)[3:5]
    norm = FP32LayerNorm(hidden, EPS, elementwise_affine=False).cuda()
    with torch.inference_mode():
        eager_res = residual + x
        eager_out = (norm(eager_res) * (1.0 + scale) + shift).to(torch.bfloat16)
        assert exact.supports_residual_norm(residual, x, 1, shift, scale)
        res16, sum32 = exact.gate_residual(residual, x, 1, fp32_copy=True)
        assert _same_bits(eager_res, res16)
        assert _same_bits(eager_res.float(), sum32)
        out, res = exact.residual_norm_modulate(residual, x, 1, shift, scale, norm)
        assert _same_bits(eager_out, out)
        assert _same_bits(eager_res, res)


@cuda_only
@pytest.mark.parametrize("batch,seq,hidden,per_token", SHAPES)
def test_site4_mlp_residual(batch, seq, hidden, per_token):
    """Wan feed-forward residual: ``(residual + ff * gate).to(orig_dtype)``."""
    residual, x = _stream(batch, seq, hidden, seed=4)
    gate = _modulation(batch, seq, hidden, per_token, seed=4)[5]
    with torch.inference_mode():
        eager = (residual + x * gate).to(torch.bfloat16)
        assert exact.supports_gate_residual(residual, x, gate)
        out, fp32_copy = exact.gate_residual(residual, x, gate, fp32_copy=False)
        assert fp32_copy is None
        assert _same_bits(eager, out)


@cuda_only
def test_scalar_gate_without_modulation_returns_eager_dtype():
    """gate 1 and no modulation: eager's FP32LayerNorm returns bf16, so the exact path must too."""
    residual, x = _stream(1, 512, 1536, seed=5)
    norm = FP32LayerNorm(1536, EPS, elementwise_affine=False).cuda()
    with torch.inference_mode():
        eager = norm(residual + x)
        out, _ = exact.residual_norm_modulate(residual, x, 1, None, None, norm)
        assert _same_bits(eager, out)


@cuda_only
def test_unsupported_inputs_refuse():
    residual, x = _stream(1, 64, 1536, seed=6)
    fp32_gate = torch.randn(1, 1, 1536, device="cuda")
    with torch.inference_mode():
        assert exact.supports_gate_residual(residual, x, fp32_gate)
        # bf16 gate rounds the product in eager; left to eager.
        assert not exact.supports_gate_residual(residual, x, fp32_gate.to(torch.bfloat16))
        # 4-dim per-frame gate takes the unflatten path in eager.
        assert not exact.supports_gate_residual(residual, x, fp32_gate.unsqueeze(1))
        # fp32 stream keeps eager.
        assert not exact.supports_gate_residual(residual.float(), x.float(), fp32_gate)
        # Narrower modulation would broadcast in eager; the kernel indexes H columns.
        assert not exact.supports_modulate(residual.float(), fp32_gate[..., :1], fp32_gate)
        # One of shift/scale missing is the caller's error, not a fusion case.
        assert not exact.supports_residual_norm(residual, x, 1, None, fp32_gate)
    # Grad-enabled forwards have no backward here.
    assert not exact.supports_gate_residual(residual, x, fp32_gate)
    assert not exact.supports_modulate(residual.float(), fp32_gate, fp32_gate)


def test_cpu_inputs_refuse():
    residual = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    with torch.inference_mode():
        assert not exact.supports_gate_residual(residual, residual, 1)
