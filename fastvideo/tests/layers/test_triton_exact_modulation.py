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
        assert exact.supports_residual_norm(residual, x, gate, None, None, norm)
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
        assert exact.supports_residual_norm(residual, x, 1, shift, scale, norm)
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
    norm = FP32LayerNorm(1536, EPS, elementwise_affine=False).cuda()
    with torch.inference_mode():
        assert exact.supports_gate_residual(residual, x, fp32_gate)
        # A bf16 gate takes the rounded-product path; a [B, 1, 1, H] table is
        # the per-frame form with one frame.
        assert exact.supports_gate_residual(residual, x, fp32_gate.to(torch.bfloat16))
        assert exact.supports_gate_residual(residual, x, fp32_gate.unsqueeze(1))
        # fp16 tables are not an eager pattern in any Wan block.
        assert not exact.supports_gate_residual(residual, x, fp32_gate.half())
        # Five dims is neither layout.
        assert not exact.supports_gate_residual(residual, x, fp32_gate[None, None])
        # fp32 stream keeps eager.
        assert not exact.supports_gate_residual(residual.float(), x.float(), fp32_gate)
        # Narrower modulation would broadcast in eager; the kernel indexes H columns.
        assert not exact.supports_modulate(residual.float(), fp32_gate[..., :1], fp32_gate)
        # One of shift/scale missing is the caller's error, not a fusion case.
        assert not exact.supports_residual_norm(residual, x, 1, None, fp32_gate, norm)
        # RMSNorm and other non-LayerNorm modules keep their own path.
        assert not exact.supports_residual_norm(residual, x, 1, None, None, torch.nn.Identity())
    # Grad-enabled forwards have no backward here.
    assert not exact.supports_gate_residual(residual, x, fp32_gate)
    assert not exact.supports_modulate(residual.float(), fp32_gate, fp32_gate)


def test_cpu_inputs_refuse():
    residual = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    with torch.inference_mode():
        assert not exact.supports_gate_residual(residual, residual, 1)


# ---------------------------------------------------------------------------
# Causal Wan (fastvideo/models/wan/causal_transformer.py): bf16 modulation
# tables, one row per frame ([B, F, 1, H], broadcast through unflatten), a plain
# nn.LayerNorm norm1 in the stream dtype, and no caller-side casts.
# ---------------------------------------------------------------------------

CAUSAL_SHAPES = [
    pytest.param(1, 3, 1560, 1536, id="sfwan-1.3b-3frames-480p"),
    pytest.param(2, 3, 20, 1536, id="batch2-small-frames"),
    pytest.param(1, 4, 256, 3072, id="4frames-3072"),
]


def _causal_tables(batch: int, frames: int, hidden: int, seed: int, dtype=torch.bfloat16):
    """Six [B, F, 1, H] chunks of scale_shift_table + temb, as the causal block makes them."""
    torch.manual_seed(seed + 200)
    table = (torch.randn(batch, frames, 6, hidden, device="cuda") * 0.5).to(dtype)
    return tuple(table.chunk(6, dim=2))


def _frame_mul(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    frames = table.shape[1]
    return (x.unflatten(dim=1, sizes=(frames, x.shape[1] // frames)) * table).flatten(1, 2)


@cuda_only
@pytest.mark.parametrize("batch,frames,frame_seqlen,hidden", CAUSAL_SHAPES)
def test_causal_site1_bf16_norm1_per_frame_modulate(batch, frames, frame_seqlen, hidden):
    """Causal pre-self-attn: bf16 LayerNorm output, bf16 per-frame tables, no cast."""
    seq = frames * frame_seqlen
    residual, _ = _stream(batch, seq, hidden, seed=11)
    shift, scale = _causal_tables(batch, frames, hidden, seed=11)[:2]
    norm1 = torch.nn.LayerNorm(hidden, EPS, elementwise_affine=False).cuda().to(torch.bfloat16)
    with torch.inference_mode():
        eager = (norm1(residual).unflatten(dim=1, sizes=(frames, frame_seqlen)) * (1 + scale) + shift).flatten(1, 2)
        assert eager.dtype == torch.bfloat16
        normed = norm1(residual)
        assert exact.supports_modulate(normed, scale, shift)
        assert _same_bits(eager, exact.modulate(normed, scale, shift, round_input=False))
        assert _same_bits(eager, exact.modulate_to(normed, scale, shift, None))


@cuda_only
@pytest.mark.parametrize("batch,frames,frame_seqlen,hidden", CAUSAL_SHAPES)
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32], ids=["ln-bf16", "ln-fp32"])
def test_causal_site2_bf16_gate_plain_layernorm(batch, frames, frame_seqlen, hidden, norm_dtype):
    """Causal post-self-attn: bf16 per-frame gate, affine nn.LayerNorm, no modulation."""
    seq = frames * frame_seqlen
    residual, x = _stream(batch, seq, hidden, seed=12)
    gate = _causal_tables(batch, frames, hidden, seed=12)[2]
    norm = torch.nn.LayerNorm(hidden, EPS, elementwise_affine=True, dtype=torch.float32).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden) * 0.02 + 1.0)
        norm.bias.copy_(torch.randn(hidden) * 0.01)
    norm = norm.to(norm_dtype)
    with torch.inference_mode():
        eager_res = residual + _frame_mul(x, gate)
        assert eager_res.dtype == torch.bfloat16
        try:
            eager_normed = norm(eager_res)
        except RuntimeError as err:  # eager itself refuses this param/input dtype mix
            pytest.skip(f"eager LayerNorm rejects {norm_dtype} params on bf16 input: {err}")
        assert exact.supports_residual_norm(residual, x, gate, None, None, norm)
        out, res = exact.residual_norm_modulate(residual, x, gate, None, None, norm)
        assert _same_bits(eager_res, res)
        assert _same_bits(eager_normed, out)


@cuda_only
@pytest.mark.parametrize("batch,frames,frame_seqlen,hidden", CAUSAL_SHAPES)
def test_causal_site3_residual_norm_bf16_modulate(batch, frames, frame_seqlen, hidden):
    """Causal post-cross-attn: gate 1, no-affine nn.LayerNorm, bf16 per-frame modulation, no cast."""
    seq = frames * frame_seqlen
    residual, x = _stream(batch, seq, hidden, seed=13)
    shift, scale = _causal_tables(batch, frames, hidden, seed=13)[3:5]
    norm = torch.nn.LayerNorm(hidden, EPS, elementwise_affine=False).cuda()
    with torch.inference_mode():
        eager_res = residual + x
        normalized = norm(eager_res)
        eager_out = (normalized.unflatten(dim=1, sizes=(frames, frame_seqlen)) * (1.0 + scale) + shift).flatten(1, 2)
        assert eager_out.dtype == torch.bfloat16
        assert exact.supports_residual_norm(residual, x, 1, shift, scale, norm)
        out, res = exact.residual_norm_modulate(residual, x, 1, shift, scale, norm)
        assert _same_bits(eager_res, res)
        assert _same_bits(eager_out, out)


@cuda_only
@pytest.mark.parametrize("batch,frames,frame_seqlen,hidden", CAUSAL_SHAPES)
def test_causal_site4_bf16_per_frame_gate(batch, frames, frame_seqlen, hidden):
    """Causal feed-forward residual: bf16 per-frame gate, output stays bf16."""
    seq = frames * frame_seqlen
    residual, x = _stream(batch, seq, hidden, seed=14)
    gate = _causal_tables(batch, frames, hidden, seed=14)[5]
    with torch.inference_mode():
        eager = residual + _frame_mul(x, gate)
        assert exact.supports_gate_residual(residual, x, gate)
        assert _same_bits(eager, exact.gate_residual(residual, x, gate, fp32_copy=False)[0])


@cuda_only
def test_fp32_per_frame_tables_match_dense_chain():
    """fp32 tables with one row per frame take the dense (fp32-promoting) rounding points."""
    batch, frames, frame_seqlen, hidden = 2, 3, 40, 1536
    seq = frames * frame_seqlen
    residual, x = _stream(batch, seq, hidden, seed=15)
    shift, scale, gate = _causal_tables(batch, frames, hidden, seed=15, dtype=torch.float32)[:3]
    norm = FP32LayerNorm(hidden, EPS, elementwise_affine=True).cuda()
    with torch.inference_mode():
        eager_sum = residual + _frame_mul(x, gate)
        assert eager_sum.dtype == torch.float32
        eager_normed = norm(eager_sum)
        eager_mod = (eager_normed.unflatten(1, (frames, frame_seqlen)) * (1.0 + scale) + shift).flatten(1, 2)
        out, res = exact.residual_norm_modulate(residual, x, gate, shift, scale, norm)
        assert _same_bits(eager_sum.to(torch.bfloat16), res)
        assert _same_bits(eager_mod.to(torch.bfloat16), out)
        assert _same_bits(eager_sum, exact.gate_residual(residual, x, gate, fp32_copy=True)[1])


@cuda_only
def test_per_frame_layout_refuses_bad_tables():
    residual, x = _stream(1, 12, 64, seed=16)
    good = torch.randn(1, 3, 1, 64, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert exact.supports_gate_residual(residual, x, good)
        # 12 tokens do not split into 5 frames.
        assert not exact.supports_gate_residual(residual, x,
                                                torch.randn(1, 5, 1, 64, device="cuda", dtype=torch.bfloat16))
        # The singleton token dim must be 1.
        assert not exact.supports_gate_residual(residual, x,
                                                torch.randn(1, 3, 2, 64, device="cuda", dtype=torch.bfloat16))
        # Mixed table dtypes are not an eager pattern.
        assert not exact.supports_modulate(residual.float(), good, good.float())


@cuda_only
@pytest.mark.parametrize("batch,frames,frame_seqlen,hidden", CAUSAL_SHAPES)
def test_causal_autocast_fp32_normed_bf16_tables(batch, frames, frame_seqlen, hidden):
    """Causal under autocast: LayerNorm returns fp32, tables stay bf16, the projection casts to bf16.

    Eager rounds ``1 + scale`` to bf16 (table dtype) but keeps the product and
    the sum in fp32; the explicit cast in the block is that projection's cast.
    """
    seq = frames * frame_seqlen
    residual, x = _stream(batch, seq, hidden, seed=17)
    shift, scale, gate = _causal_tables(batch, frames, hidden, seed=17)[:3]
    norm1 = torch.nn.LayerNorm(hidden, EPS, elementwise_affine=False).cuda().to(torch.bfloat16)
    norm3 = torch.nn.LayerNorm(hidden, EPS, elementwise_affine=False).cuda().to(torch.bfloat16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        normed = norm1(residual)
        assert normed.dtype == torch.float32
        eager1 = (normed.unflatten(1, (frames, frame_seqlen)) * (1 + scale) + shift).flatten(1, 2)
        assert eager1.dtype == torch.float32
        assert _same_bits(eager1.to(torch.bfloat16), exact.modulate(normed, scale, shift, round_input=False))
        assert _same_bits(eager1.to(torch.bfloat16), exact.modulate_to(normed, scale, shift, torch.bfloat16))
        # fp32 result with no cast requested: the eager expression is returned as is.
        assert _same_bits(eager1, exact.modulate_to(normed, scale, shift, None))

        eager_res = residual + x
        eager3 = (norm3(eager_res).unflatten(1, (frames, frame_seqlen)) * (1.0 + scale) + shift).flatten(1, 2)
        out, res = exact.residual_norm_modulate(residual, x, 1, shift, scale, norm3)
        assert _same_bits(eager_res, res)
        assert _same_bits(eager3.to(torch.bfloat16), out)

        eager_sum = residual + _frame_mul(x, gate)
        normed2 = norm3(eager_sum)
        out2, res2 = exact.residual_norm_modulate(residual, x, gate, None, None, norm3)
        assert _same_bits(eager_sum, res2)
        assert _same_bits(normed2, out2)
