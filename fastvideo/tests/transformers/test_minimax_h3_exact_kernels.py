# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3's bit-exact Triton kernels must equal the eager code they replace, bit for bit.

Unlike the Sol-Engine fusion tests (``test_minimax_h3_*_fusion.py``), which
allow a tolerance, every GPU comparison here is ``torch.equal`` on the raw
BF16 bit patterns.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention, _enabled_minimax_h3_exact_kernels
from fastvideo.models.dits.minimax_h3_fusions import exact


def _require_cuda_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 exact kernels require CUDA")
    if not exact.HAVE_TRITON:
        pytest.skip("MiniMax H3 exact kernels require Triton")


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _bf16(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape, device="cuda") * scale).to(torch.bfloat16)


@pytest.fixture
def inference():
    torch.manual_seed(0)
    with torch.inference_mode():
        yield


def test_exact_kernel_set_parsing() -> None:
    assert _enabled_minimax_h3_exact_kernels("") == frozenset()
    assert _enabled_minimax_h3_exact_kernels("none") == frozenset()
    assert _enabled_minimax_h3_exact_kernels("all") == frozenset({"rope", "modulate", "swiglu"})
    assert _enabled_minimax_h3_exact_kernels(" rope, swiglu ") == frozenset({"rope", "swiglu"})
    with pytest.raises(ValueError, match="qknorm_rope"):
        _enabled_minimax_h3_exact_kernels("qknorm_rope")


def test_exact_kernels_refuse_grad_and_cpu_inputs() -> None:
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    assert not exact.supports_rowwise(x)
    with torch.no_grad():
        assert not exact.supports_rowwise(x)


@pytest.mark.gpu
@pytest.mark.parametrize("seq", [777, 9000])
def test_modulate_and_gated_residual_are_bit_exact(inference, seq: int) -> None:
    _require_cuda_triton()
    hidden, rows = 6144, 24
    index = torch.randint(0, rows, (seq, ), device="cuda")
    # Six chunks of one AdaLN projection: strided views, as the DiT produces them.
    shift, scale, gate = _bf16(rows, 6 * hidden, scale=0.5).chunk(6, dim=-1)[:3]
    normed, residual, branch = _bf16(1, seq, hidden), _bf16(1, seq, hidden, scale=4), _bf16(1, seq, hidden)
    assert exact.supports_rowwise(normed, shift, scale, gate)
    eager = normed * (1.0 + scale.index_select(0, index)) + shift.index_select(0, index)
    assert _same_bits(eager, exact.modulate(normed, scale, shift, index))
    eager = residual + gate.index_select(0, index) * branch
    assert _same_bits(eager, exact.gate_residual(residual, gate, branch, index))


@pytest.mark.gpu
def test_modulate_accepts_tables_with_different_row_strides(inference) -> None:
    _require_cuda_triton()
    hidden, rows, seq = 1024, 8, 300
    index = torch.randint(0, rows, (seq, ), device="cuda")
    scale = _bf16(rows, 3 * hidden).chunk(3, dim=-1)[1]
    shift = _bf16(rows, hidden)
    normed = _bf16(1, seq, hidden)
    eager = normed * (1.0 + scale.index_select(0, index)) + shift.index_select(0, index)
    assert _same_bits(eager, exact.modulate(normed, scale, shift, index))


@pytest.mark.gpu
def test_swiglu_is_bit_exact_including_edge_values(inference) -> None:
    _require_cuda_triton()
    ffn = 16384
    packed = _bf16(1, 777, 2 * ffn, scale=4)
    packed[0, 0, ffn:ffn + 8] = torch.tensor([0.0, -0.0, 88.0, -88.0, 100.0, -100.0, 1e-30, -1e-30])
    value, gate = packed.chunk(2, dim=-1)
    assert _same_bits(value * F.silu(gate), exact.swiglu(packed))


@pytest.mark.gpu
@pytest.mark.parametrize("seq", [777, 9000])
def test_rope_prefix_matches_apply_rotary_emb_bit_for_bit(inference, seq: int) -> None:
    _require_cuda_triton()
    # H3 geometry: 128-channel heads with a 96-channel rotary prefix.
    x = _bf16(1, seq, 48, 128, scale=3)
    x[0, :5], x[0, 5:10] = 0.0, -0.0
    positions = torch.arange(seq, device="cuda", dtype=torch.float32)[:, None]
    freqs = positions * torch.logspace(0, -4, 48, device="cuda")[None]
    freqs = torch.cat((freqs, freqs), dim=-1)
    rotary_emb = (freqs.cos(), freqs.sin())
    assert exact.supports_rope(x, rotary_emb)
    assert _same_bits(MiniMaxH3Attention._apply_rotary_emb(x, rotary_emb), exact.rope_prefix(x, rotary_emb))
