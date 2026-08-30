# SPDX-License-Identifier: Apache-2.0
"""Parity and dispatch contracts for affine dequant + dense GEMM."""

from __future__ import annotations

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for affine dq-GEMM tests")

from fastvideo.mlx_runtime.fastwan import (  # noqa: E402
    MLXQuantizationSpec,
    affine_dq_gemm_min_m,
    dq_gemm_engaged,
    linear,
    quantize_matrix,
    reset_dq_gemm_telemetry,
)
from fastvideo.mlx_runtime.minimax_h3 import linear as h3_linear  # noqa: E402

AFFINE_BITS = (2, 3, 4, 5, 6, 8)
GROUP_SIZES = (32, 64, 128)


def _qmm(x, weight):
    return mx.quantized_matmul(
        x,
        weight.weight,
        weight.scales,
        weight.biases,
        transpose=True,
        group_size=weight.spec.group_size,
        bits=weight.spec.bits,
        mode=weight.spec.mode,
    ).astype(x.dtype)


def _try_quantize(out_features: int, in_features: int, bits: int, group_size: int):
    spec = MLXQuantizationSpec(mode="affine", bits=bits, group_size=group_size)
    weight = mx.random.normal((out_features, in_features)).astype(mx.bfloat16)
    try:
        quantized = quantize_matrix(weight, spec)
        mx.eval(quantized.weight, quantized.scales, quantized.biases)
        return quantized
    except Exception as exc:  # noqa: BLE001 - MLX support varies by version.
        pytest.skip(f"affine bits={bits} group_size={group_size} unsupported: {exc}")


def _rel_l2(a, b) -> float:
    left = np.asarray(a.astype(mx.float32))
    right = np.asarray(b.astype(mx.float32))
    denom = max(float(np.linalg.norm(left)), 1e-12)
    return float(np.linalg.norm(left - right) / denom)


@pytest.mark.parametrize("bits", AFFINE_BITS)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_dq_gemm_matches_qmm_for_supported_bit_widths(bits: int, group_size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_MLX_DQ_GEMM", "8")
    reset_dq_gemm_telemetry()
    in_features = group_size * 4
    out_features = group_size * 2
    quantized = _try_quantize(out_features, in_features, bits, group_size)
    x = mx.random.normal((16, in_features)).astype(mx.bfloat16)
    mx.eval(x)
    before = dq_gemm_engaged()
    got = h3_linear(x, quantized)
    ref = _qmm(x, quantized)
    mx.eval(got, ref)
    assert dq_gemm_engaged() == before + 1
    rel = _rel_l2(got, ref)
    ref_np = np.asarray(ref.astype(mx.float32))
    got_np = np.asarray(got.astype(mx.float32))
    scale = max(float(np.max(np.abs(ref_np))), 1e-3)
    assert rel < 2e-2, rel
    assert float(np.max(np.abs(got_np - ref_np))) / scale < 0.08


def test_dq_gemm_with_bias_and_batched_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_MLX_DQ_GEMM", "4")
    reset_dq_gemm_telemetry()
    quantized = _try_quantize(64, 128, bits=6, group_size=64)
    x = mx.random.normal((2, 8, 128)).astype(mx.bfloat16)
    bias = mx.random.normal((64, )).astype(mx.bfloat16)
    mx.eval(x, bias)
    got = h3_linear(x, quantized, bias)
    ref = _qmm(x, quantized) + bias
    mx.eval(got, ref)
    assert dq_gemm_engaged() == 1
    assert _rel_l2(got, ref) < 2e-2
    # transpose=True contract: output last dim is out_features.
    assert got.shape == (2, 8, 64)


def test_dq_gemm_stays_on_qmm_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_MLX_DQ_GEMM", "768")
    reset_dq_gemm_telemetry()
    quantized = _try_quantize(64, 128, bits=6, group_size=64)
    x = mx.random.normal((32, 128)).astype(mx.bfloat16)
    mx.eval(x)
    got = h3_linear(x, quantized)
    ref = _qmm(x, quantized)
    mx.eval(got, ref)
    assert dq_gemm_engaged() == 0
    np.testing.assert_array_equal(np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32)))


def test_dq_gemm_env_zero_disables_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_MLX_DQ_GEMM", "0")
    assert affine_dq_gemm_min_m() is None
    reset_dq_gemm_telemetry()
    quantized = _try_quantize(64, 128, bits=6, group_size=64)
    x = mx.random.normal((1024, 128)).astype(mx.bfloat16)
    mx.eval(x)
    got = h3_linear(x, quantized)
    ref = _qmm(x, quantized)
    mx.eval(got, ref)
    assert dq_gemm_engaged() == 0
    np.testing.assert_array_equal(np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32)))


def test_shared_linear_stays_on_qmm_at_wide_m(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_MLX_DQ_GEMM", "1")
    reset_dq_gemm_telemetry()
    quantized = _try_quantize(64, 128, bits=6, group_size=64)
    x = mx.random.normal((1024, 128)).astype(mx.bfloat16)
    got = linear(x, quantized)
    ref = _qmm(x, quantized)
    mx.eval(got, ref)
    assert dq_gemm_engaged() == 0
    np.testing.assert_array_equal(np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32)))


def test_non_affine_weights_stay_on_quantized_matmul(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_MLX_DQ_GEMM", "1")
    reset_dq_gemm_telemetry()
    spec = MLXQuantizationSpec(mode="mxfp8")
    weight = mx.random.normal((64, 64)).astype(mx.bfloat16)
    try:
        quantized = quantize_matrix(weight, spec)
        mx.eval(quantized.weight, quantized.scales)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"mxfp8 unsupported: {exc}")
    x = mx.random.normal((1024, 64)).astype(mx.bfloat16)
    mx.eval(x)
    got = h3_linear(x, quantized)
    mx.eval(got)
    assert dq_gemm_engaged() == 0
    assert got.shape == (1024, 64)


def test_default_floor_is_measured_768(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_MLX_DQ_GEMM", "1")
    assert affine_dq_gemm_min_m() == 768
    monkeypatch.setenv("FASTVIDEO_MLX_DQ_GEMM", "2048")
    assert affine_dq_gemm_min_m() == 2048
    monkeypatch.delenv("FASTVIDEO_MLX_DQ_GEMM", raising=False)
    os.environ.pop("FASTVIDEO_MLX_DQ_GEMM", None)
    # Default with unset env is on at the measured floor.
    monkeypatch.delenv("FASTVIDEO_MLX_DQ_GEMM", raising=False)
    if "FASTVIDEO_MLX_DQ_GEMM" in os.environ:
        pytest.skip("parent environment pinned FASTVIDEO_MLX_DQ_GEMM")
    assert affine_dq_gemm_min_m() == 768
