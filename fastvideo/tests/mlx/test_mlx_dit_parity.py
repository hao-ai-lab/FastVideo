# SPDX-License-Identifier: Apache-2.0
"""Full-DiT parity: the MLX Wan runtime vs the PyTorch reference model.

This is the M1 "trustworthy baseline" gate for the Apple Silicon path: a tiny
random-weight ``WanTransformer3DModel`` is run end to end (patch embed ->
condition -> transformer blocks -> unpatchify) in PyTorch and in
``fastvideo.mlx_runtime.fastwan.MLXWanDiT`` with identical weights, and the
outputs must match within pinned fp32 tolerances.

Runs anywhere MLX is installed: on Apple Silicon it exercises the Metal
device, on Linux/CI it runs on MLX's CPU backend (``pip install 'mlx[cpu]'``)
-- the graph is identical, so CPU parity is the CI-friendly golden variant.

    pytest fastvideo/tests/mlx/test_mlx_dit_parity.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for DiT parity tests")

from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec  # noqa: E402
from fastvideo.tests.mlx.tiny_wan import (  # noqa: E402
    build_hf_config,
    build_inputs,
    build_tiny_wan_config,
    build_torch_model,
    mlx_dit_from_torch_model,
    mlx_output,
    mlx_rotary_embeddings,
    torch_reference_output,
)

# Pinned fp32 tolerances for the full forward (patch embed through unpatchify),
# matching the single-block parity example. Measured max_abs_diff on the MLX
# CPU backend is ~1.7e-6, so 2e-4 keeps ~100x headroom for accumulation-order
# differences across MLX backends while still catching real math/layout bugs.
FP32_ATOL = 2e-4
FP32_RTOL = 2e-4

# Quantized inference is lossy by design; gate it on signal-to-noise vs the
# fp32 MLX output instead of elementwise closeness. int8 (group size 64) on
# this tiny model measures ~43 dB; 20 dB leaves headroom without letting a
# broken dequant path (which lands near 0 dB) slip through.
INT8_MIN_SNR_DB = 20.0


def test_full_dit_forward_matches_torch_reference(distributed_setup) -> None:
    model = build_torch_model()
    hidden_states, encoder_hidden_states, timestep = build_inputs()

    torch_out = torch_reference_output(model, hidden_states, encoder_hidden_states, timestep)

    dit = mlx_dit_from_torch_model(model, build_hf_config(build_tiny_wan_config()))
    mlx_out = mlx_output(dit, hidden_states, encoder_hidden_states, timestep, mlx_rotary_embeddings(hidden_states))

    assert mlx_out.shape == torch_out.shape
    assert np.isfinite(mlx_out).all()
    max_abs = float(np.abs(torch_out - mlx_out).max())
    assert np.allclose(torch_out, mlx_out, atol=FP32_ATOL, rtol=FP32_RTOL), (
        f"MLX full-DiT forward diverged from the torch reference: max_abs_diff={max_abs:.3e} "
        f"(atol={FP32_ATOL}, rtol={FP32_RTOL})")


def test_full_dit_forward_int8_stays_close_to_fp32(distributed_setup) -> None:
    model = build_torch_model()
    hidden_states, encoder_hidden_states, timestep = build_inputs()
    hf_config = build_hf_config(build_tiny_wan_config())
    freqs_cis = mlx_rotary_embeddings(hidden_states)

    fp32_out = mlx_output(
        mlx_dit_from_torch_model(model, hf_config), hidden_states, encoder_hidden_states, timestep, freqs_cis)
    int8_out = mlx_output(
        mlx_dit_from_torch_model(model, hf_config, quantization=MLXQuantizationSpec.from_name("int8")),
        hidden_states, encoder_hidden_states, timestep, freqs_cis)

    assert np.isfinite(int8_out).all()
    noise = float(np.mean(np.square(int8_out - fp32_out)))
    signal = float(np.mean(np.square(fp32_out)))
    snr_db = 10.0 * np.log10(signal / noise) if noise > 0 else float("inf")
    assert snr_db >= INT8_MIN_SNR_DB, (
        f"int8-quantized MLX DiT output is too far from fp32: SNR {snr_db:.1f} dB < {INT8_MIN_SNR_DB} dB")
