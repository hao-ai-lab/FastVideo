# SPDX-License-Identifier: Apache-2.0
"""The M4 Phase A gate: the torch fake-quant twin must reproduce MLX's affine
quantizer before any GPU is spent on a quantization-aware-training run.

If train-time fake-quantization differs from ``mx.quantize``/``mx.dequantize``
at deploy time, the QAT gains evaporate. But "differs" has to be defined against
the right thing. MLX's *CPU* and *Metal* kernels do not agree bit-for-bit: the
CPU kernel evaluates the affine math in the input's fp16, while the Metal kernel
accumulates in fp32. So this gate is two assertions, not one:

(a) **Decisions, bit-pinned on the CPU stream.** The integer codes, scales, and
    biases — the quantizer's actual decisions, and what a QAT model learns to be
    robust to — match MLX's CPU stream exactly. The CPU stream is the canonical,
    deterministic, machine-independent specification of the affine algorithm;
    pinning here catches the real regressions (wrong rounding rule, wrong
    zero-point anchoring, wrong group reduction) reproducibly on any hardware.

(b) **Deploy reconstruction, tolerance-pinned on the Metal stream.** The Mac
    deploy runtime runs on Metal, where fp32 accumulation makes the codes flip
    on a vanishing fraction of boundary weights and the dequantized values drift
    slightly. That drift is measured and bounded here with recorded headroom,
    and the real deploy op (``mx.quantized_matmul``) is checked end-to-end. All
    of it sits four orders of magnitude under the error the matmul already
    tolerates, so the twin faithfully models deploy.

Measured headroom (M4 Max, MLX 0.31.2; see apple_silicon_benchmark_baseline.md):
Metal code-flip rate 0.0147% (all +/-1 LSB), dequant accumulation drift ~1.2e-4,
quantized_matmul vs twin max-abs 1.1e-3 against a 2e-2 deploy tolerance (18x).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core", reason="MLX is required for QAT numerics parity tests")

from fastvideo.layers.quantization.mlx_affine_qat import (  # noqa: E402
    fake_quantize_mlx_affine,
    mlx_affine_dequantize_reference,
    mlx_affine_quantize_reference,
)

# --- Metal deploy-drift budgets (measured values + margin) -------------------
# Codes are backend-stable except for boundary weights that fp32-vs-fp16
# rounding tips by one bin. Measured 0.0147% over 156k elements, all +/-1 LSB.
METAL_CODE_FLIP_RATE_BUDGET = 1e-3
# Dequant drift on elements whose code did NOT flip is pure fp32-vs-fp16
# accumulation of ``code * scale + bias``. Measured max 1.2e-4 for fp16.
METAL_ACCUM_DRIFT_BUDGET = 5e-4
# Real deploy op headroom: ``mx.quantized_matmul`` vs the fake-quant linear.
QMATMUL_DEPLOY_TOL = 2e-2


def _unpack_uint32_codes(packed: np.ndarray, *, bits: int, out_cols: int) -> np.ndarray:
    """Unpack MLX's little-endian uint32 words into per-element integer codes."""
    el_per_word = 32 // bits
    bitmask = (1 << bits) - 1
    words = packed.astype(np.uint64)
    codes = np.zeros((*packed.shape[:-1], packed.shape[-1] * el_per_word), dtype=np.int32)
    for k in range(el_per_word):
        codes[..., k::el_per_word] = ((words >> (k * bits)) & bitmask).astype(np.int32)
    return codes[..., :out_cols]


def _mlx_quantize(w_np: np.ndarray, *, group_size: int, bits: int, device) -> tuple:
    """Quantize/dequantize on an explicit MLX stream (CPU = spec, GPU = deploy)."""
    with mx.stream(device):
        q, scales, biases = mx.quantize(mx.array(w_np), group_size=group_size, bits=bits, mode="affine")
        deq = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits, mode="affine")
        mx.eval(q, scales, biases, deq)
    return (np.array(q), np.array(scales), np.array(biases), np.array(deq))


# --- (a) Decisions: bit-pinned against MLX's CPU stream ----------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("shape", [(4, 128), (3, 64), (16, 256)])
def test_quantizer_decisions_bitmatch_mlx_cpu(dtype: torch.dtype, bits: int, shape: tuple[int, int]) -> None:
    """Codes, scales, and biases match MLX's CPU kernel bit-for-bit — the
    canonical, machine-independent definition of the quantizer's decisions."""
    torch.manual_seed(1234)
    w = torch.randn(*shape, dtype=torch.float32).to(dtype) * 0.05

    codes, scales, biases = mlx_affine_quantize_reference(w, group_size=64, bits=bits)

    w_np = w.float().numpy() if dtype == torch.float32 else w.numpy()
    q_mlx, scales_mlx, biases_mlx, _ = _mlx_quantize(w_np, group_size=64, bits=bits, device=mx.cpu)

    codes_mlx = _unpack_uint32_codes(q_mlx, bits=bits, out_cols=shape[-1])
    codes_flat = codes.reshape(shape[0], -1).numpy()

    np.testing.assert_array_equal(codes_flat, codes_mlx)
    np.testing.assert_array_equal(scales.float().numpy(), np.asarray(scales_mlx, dtype=np.float32))
    np.testing.assert_array_equal(biases.float().numpy(), np.asarray(biases_mlx, dtype=np.float32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dequant_bitmatch_mlx_cpu(dtype: torch.dtype) -> None:
    """The twin's dequantized reconstruction matches MLX's CPU kernel exactly."""
    torch.manual_seed(7)
    w = torch.randn(8, 192, dtype=torch.float32).to(dtype) * 0.05

    codes, scales, biases = mlx_affine_quantize_reference(w, group_size=64, bits=8)
    deq = mlx_affine_dequantize_reference(codes, scales, biases, out_shape=w.shape)

    w_np = w.float().numpy() if dtype == torch.float32 else w.numpy()
    _, _, _, deq_mlx = _mlx_quantize(w_np, group_size=64, bits=8, device=mx.cpu)

    np.testing.assert_array_equal(deq.float().numpy(), np.asarray(deq_mlx, dtype=np.float32))


def test_fake_quantize_matches_cpu_spec_and_passes_gradients() -> None:
    """The QAT forward equals the CPU-spec reconstruction (deterministic), and
    the straight-through estimator passes gradients unchanged."""
    torch.manual_seed(99)
    # bf16 master weights, exactly like a QAT training run.
    w = (torch.randn(4, 128, dtype=torch.float32) * 0.05).to(torch.bfloat16).requires_grad_(True)

    fq = fake_quantize_mlx_affine(w, group_size=64, bits=8, simulate_dtype=torch.float16)

    # Forward: quantizing the fp16-cast weight (the loader casts checkpoints to
    # fp16 before mx.quantize) reproduces MLX's CPU dequant bit-for-bit.
    w_fp16 = w.detach().to(torch.float16)
    _, _, _, deq_mlx = _mlx_quantize(w_fp16.numpy(), group_size=64, bits=8, device=mx.cpu)
    np.testing.assert_array_equal(
        fq.detach().to(torch.float16).float().numpy(),
        np.asarray(deq_mlx, dtype=np.float32),
    )

    # Backward: straight-through — gradients reach the master weight unchanged.
    fq.sum().backward()
    assert w.grad is not None
    np.testing.assert_array_equal(w.grad.float().numpy(), np.ones_like(w.grad.float().numpy()))


# --- (b) Deploy reconstruction: tolerance-pinned against the Metal stream -----


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal deploy-drift gate needs a Metal device")
@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("shape", [(16, 256), (128, 256)])
def test_metal_deploy_drift_within_budget(bits: int, shape: tuple[int, int]) -> None:
    """On Metal (the deploy device) fp32 accumulation flips a tiny fraction of
    boundary codes and drifts the reconstruction; both stay within budget."""
    torch.manual_seed(1234)
    w = (torch.randn(*shape, dtype=torch.float32).to(torch.float16)) * 0.05

    codes, scales, biases = mlx_affine_quantize_reference(w, group_size=64, bits=bits)
    deq_ref = mlx_affine_dequantize_reference(codes, scales, biases, out_shape=w.shape).float().numpy()
    codes_ref = codes.reshape(shape[0], -1).numpy()

    q_mlx, _, _, deq_mlx = _mlx_quantize(w.numpy(), group_size=64, bits=bits, device=mx.gpu)
    codes_mlx = _unpack_uint32_codes(q_mlx, bits=bits, out_cols=shape[-1])
    deq_mlx = np.asarray(deq_mlx, dtype=np.float32)

    # Code flips are rare and only ever move by a single bin.
    flipped = codes_ref != codes_mlx
    flip_rate = float(flipped.mean())
    assert flip_rate <= METAL_CODE_FLIP_RATE_BUDGET, f"Metal code-flip rate {flip_rate:.4%} over budget"
    if flipped.any():
        assert int(np.abs(codes_ref - codes_mlx).max()) == 1, "Metal code flips must be +/-1 LSB"

    # Where the code agrees, the only difference is fp32-vs-fp16 accumulation.
    agree = ~flipped
    if agree.any():
        accum_drift = float(np.abs(deq_mlx[agree] - deq_ref[agree]).max())
        assert accum_drift <= METAL_ACCUM_DRIFT_BUDGET, f"Metal accumulation drift {accum_drift:.2e} over budget"


@pytest.mark.skipif(not mx.metal.is_available(), reason="deploy quantized_matmul runs on Metal")
def test_quantized_matmul_close_to_fake_quant_linear() -> None:
    """The real deploy op (``mx.quantized_matmul`` on Metal) tracks the
    fake-quant linear within the deploy tolerance (measured ~1.1e-3 << 2e-2)."""
    torch.manual_seed(3)
    w = (torch.randn(128, 256, dtype=torch.float32) * 0.05).to(torch.float16)
    x = (torch.randn(2, 256, dtype=torch.float32) * 0.5).to(torch.float16)

    codes, scales, biases = mlx_affine_quantize_reference(w, group_size=64, bits=8)
    deq = mlx_affine_dequantize_reference(codes, scales, biases, out_shape=w.shape)
    y_torch = (x.float() @ deq.float().T)

    with mx.stream(mx.gpu):
        q, s, b = mx.quantize(mx.array(w.numpy()), group_size=64, bits=8, mode="affine")
        y_mlx = mx.quantized_matmul(
            mx.array(x.numpy()), q, s, b, transpose=True, group_size=64, bits=8, mode="affine")
        mx.eval(y_mlx)

    # Accumulation order differs between frameworks, so this is a tolerance
    # check (the weight decisions themselves are pinned bitwise above).
    np.testing.assert_allclose(
        y_torch.numpy(), np.array(y_mlx).astype(np.float32), atol=QMATMUL_DEPLOY_TOL, rtol=QMATMUL_DEPLOY_TOL)


def test_indivisible_group_size_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        mlx_affine_quantize_reference(torch.randn(4, 100), group_size=64, bits=8)
