# SPDX-License-Identifier: Apache-2.0
"""The FP4 prefix split's row bookkeeping and dense prefix attention.

The split takes the dense prefix query tiles (text, reference, audio rows) off
the FP4 sparse kernel and attends them densely at 8-bit Q/K. These tests cover
the parts that do not need a Blackwell GPU: the FP8 rounding, the valid-key
vector the kernel's half counts expand to, and the dense attention itself -
checked against the training emulator, which is the numeric spec.
"""

from __future__ import annotations

import pytest
import torch

from fastvideo.models.dits.minimax_h3_vsa_fp4 import (_fp8_per_token, _half_row_valid, _prefix_dense_attention,
                                                      _resolve_prefix_mode)

HALF = 64


def _emulator_fp8(x: torch.Tensor) -> torch.Tensor:
    """The emulator's per-token E4M3 fake quant; it needs Triton to import."""
    try:
        from fastvideo_kernel.triton_kernels.attn_qat_vsa_train import fp8_per_token_fake_quant
    except (ImportError, KeyError):  # KeyError: a failed fastvideo_kernel import left a None entry
        pytest.skip("the Triton emulator is unavailable")
    return fp8_per_token_fake_quant(x)


def _reference_dense_attention(q, k, v, valid, scale):
    """fp32 softmax attention over ``[B, L, H, D]`` with the invalid keys masked out."""
    scores = torch.einsum("blhd,bkhd->bhlk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~valid[None, None, None, :], float("-inf"))
    return torch.einsum("bhlk,bkhd->blhd", torch.softmax(scores, dim=-1), v.float()).to(v.dtype)


def _case(rows=1, prefix=8, kv_len=16, heads=2, dim=128, tail_invalid=4):
    torch.manual_seed(0)
    q = torch.randn(rows, prefix, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(rows, kv_len, heads, dim, dtype=torch.bfloat16)
    v = torch.randn(rows, kv_len, heads, dim, dtype=torch.bfloat16)
    valid = torch.ones(kv_len, dtype=torch.bool)
    valid[kv_len - tail_invalid:] = False
    return q, k, v, valid, dim


def test_fp8_per_token_rounds_within_one_e4m3_step() -> None:
    """FP8 E4M3 has 3 mantissa bits, so a per-row scale bounds the error at ~6%."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 5, 128, dtype=torch.bfloat16)
    y = _fp8_per_token(x)
    assert y.shape == x.shape and y.dtype == torch.bfloat16
    assert torch.allclose(y, x, rtol=0.1, atol=1e-2)
    # Per (row, head) scale: a row whose magnitude is 1000x the others keeps its own precision.
    x[0, 0, 0] *= 1000.0
    assert ((_fp8_per_token(x)[0, 0, 0] - x[0, 0, 0]).norm() / x[0, 0, 0].norm()).item() < 0.1


def test_fp8_per_token_keeps_zero_rows_zero() -> None:
    """Pad rows stay exactly zero, which the kernel's zero-padded k/v relies on."""
    x = torch.zeros(2, 2, 128, dtype=torch.bfloat16)
    assert torch.equal(_fp8_per_token(x), x)


def test_fp8_per_token_matches_the_emulator() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 16, 2, 128, dtype=torch.bfloat16)
    assert torch.equal(_fp8_per_token(x), _emulator_fp8(x).to(torch.bfloat16))


def test_half_row_valid_expands_counts_to_rows() -> None:
    kv_valid = torch.tensor([HALF, 3, 0, HALF], dtype=torch.int32)
    valid = _half_row_valid(kv_valid, HALF)
    assert valid.shape == (4 * HALF, )
    assert valid.sum().item() == 2 * HALF + 3
    assert valid[:HALF].all() and not valid[HALF + 3:HALF + HALF].any()
    assert not valid[2 * HALF:3 * HALF].any()


@pytest.mark.parametrize(("mode", "steps", "step", "expected"), [
    ("fp8", -1, 0, "fp8"),
    ("fp8", -1, 7, "fp8"),
    ("bf16", 3, 2, "bf16"),
    ("fp8", 3, 3, ""),
    ("fp8", 0, 0, ""),
    ("", 4, 0, ""),
])
def test_prefix_mode_step_cutoff(mode: str, steps: int, step: int, expected: str) -> None:
    """A non-negative step limit applies the split only to the leading steps."""
    assert _resolve_prefix_mode(mode, steps, step) == expected


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
def test_prefix_dense_attention_matches_the_emulator_numerics(precision: str) -> None:
    """Smooth K, then round it - the order the emulator and the student use."""
    q, k, v, valid, dim = _case()

    out = _prefix_dense_attention(q, k, v, q.shape[1], k.shape[1], valid, precision)

    assert out.shape == (1, 2, 8, dim) and out.dtype == torch.bfloat16
    k_ref = k - k.mean(dim=1, keepdim=True)
    q_ref = q
    if precision == "fp8":
        q_ref, k_ref = _emulator_fp8(q), _emulator_fp8(k_ref)
    expected = _reference_dense_attention(q_ref, k_ref, v, valid, dim**-0.5).transpose(1, 2)

    assert (out.float() - expected.float()).norm() / expected.float().norm() < 2e-2


def test_smoothing_happens_before_the_fp8_rounding() -> None:
    """Rounding before smoothing gives different keys, so the order is load-bearing."""
    q, k, v, valid, _ = _case()
    smooth_then_round = _emulator_fp8(k - k.mean(dim=1, keepdim=True))
    round_then_smooth = _emulator_fp8(k) - _emulator_fp8(k).mean(dim=1, keepdim=True)
    assert not torch.equal(smooth_then_round, round_then_smooth)

    out = _prefix_dense_attention(q, k, v, q.shape[1], k.shape[1], valid, "fp8")
    expected = _reference_dense_attention(q, smooth_then_round, v, valid, k.shape[-1]**-0.5).transpose(1, 2)
    wrong = _reference_dense_attention(q, round_then_smooth, v, valid, k.shape[-1]**-0.5).transpose(1, 2)

    error = (out.float() - expected.float()).norm()
    assert error < (out.float() - wrong.float()).norm()


def test_prefix_dense_attention_ignores_invalid_keys() -> None:
    """Masked-out keys must not reach the output, however extreme they are."""
    torch.manual_seed(0)
    rows, prefix, valid_len, heads, dim = 1, 4, 6, 2, 128
    q = torch.randn(rows, prefix, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(rows, valid_len + 2, heads, dim, dtype=torch.bfloat16)
    v = torch.randn(rows, valid_len + 2, heads, dim, dtype=torch.bfloat16)
    valid = torch.ones(valid_len + 2, dtype=torch.bfloat16).bool()
    valid[valid_len:] = False

    masked = _prefix_dense_attention(q, k, v, prefix, valid_len + 2, valid, "bf16")

    # Truncating to the valid rows changes only the smoothing mean, an exact shift.
    k[:, valid_len:] = 0.0
    v[:, valid_len:] = 0.0
    k = k - (k[:, :valid_len].mean(dim=1, keepdim=True) - k.mean(dim=1, keepdim=True))
    truncated = _prefix_dense_attention(q, k[:, :valid_len], v[:, :valid_len], prefix, valid_len,
                                        torch.ones(valid_len, dtype=torch.bool), "bf16")

    assert (masked.float() - truncated.float()).norm() / truncated.float().norm() < 5e-3
