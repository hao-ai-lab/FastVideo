# SPDX-License-Identifier: Apache-2.0
"""MLX quantization-mode capability detection.

The affine int8/int4 modes work on every MLX build the repo supports; the
mxfp8/mxfp4/nvfp4 mode strings only exist on newer MLX builds. These tests pin
the contract: supported modes probe clean, and unsupported ones surface as
``UnsupportedMLXQuantizationError`` with an actionable message *before* any
model weights are loaded.

Runs on any MLX backend (Metal or ``mlx[cpu]``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core", reason="MLX is required for quantization capability tests")

from fastvideo.mlx_runtime.fastwan import (  # noqa: E402
    MLXQuantizationSpec,
    UnsupportedMLXQuantizationError,
    _QUANT_SUPPORT_CACHE,
    ensure_quantization_supported,
    quantization_support_error,
)


def test_affine_modes_are_supported_on_any_mlx_build() -> None:
    for name in ("int8", "int4"):
        spec = MLXQuantizationSpec.from_name(name)
        assert quantization_support_error(spec) is None, f"{name} must be supported on every MLX build"
        ensure_quantization_supported(spec)  # must not raise


def test_none_spec_is_always_supported() -> None:
    ensure_quantization_supported(None)  # must not raise


def test_probe_result_is_cached_per_spec() -> None:
    spec = MLXQuantizationSpec.from_name("int8")
    quantization_support_error(spec)
    key = (spec.mode, spec.bits, spec.group_size)
    assert key in _QUANT_SUPPORT_CACHE


def test_unsupported_mode_raises_with_actionable_message(monkeypatch) -> None:
    spec = MLXQuantizationSpec.from_name("nvfp4")
    key = (spec.mode, spec.bits, spec.group_size)
    monkeypatch.setitem(_QUANT_SUPPORT_CACHE, key, "ValueError: [quantize] Unknown mode 'nvfp4'")

    with pytest.raises(UnsupportedMLXQuantizationError) as excinfo:
        ensure_quantization_supported(spec)
    message = str(excinfo.value)
    assert "nvfp4" in message
    assert "int8" in message  # points the user at the reliable fallback


def test_loader_rejects_unsupported_mode_before_reading_weights(monkeypatch, tmp_path) -> None:
    from fastvideo.mlx_runtime.fastwan import mlx_dit_from_diffusers_safetensors

    spec_key = ("mxfp4", None, None)
    monkeypatch.setitem(_QUANT_SUPPORT_CACHE, spec_key, "ValueError: [quantize] Unknown mode 'mxfp4'")

    config_path = tmp_path / "config.json"
    config_path.write_text('{"num_layers": 1, "num_attention_heads": 4, "attention_head_dim": 16, '
                           '"ffn_dim": 128, "in_channels": 16, "out_channels": 16, '
                           '"patch_size": [1, 2, 2], "freq_dim": 64, "eps": 1e-6}')
    # The checkpoint path deliberately does not exist: the capability check
    # must fire before the loader ever opens the safetensors file.
    with pytest.raises(UnsupportedMLXQuantizationError):
        mlx_dit_from_diffusers_safetensors(tmp_path / "missing.safetensors", config_path, quantization="mxfp4")
