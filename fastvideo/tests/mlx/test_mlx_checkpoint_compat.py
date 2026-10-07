# SPDX-License-Identifier: Apache-2.0
"""Guard: NVIDIA FastWan-QAD checkpoints must not silently load on MLX."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastvideo.mlx_runtime.checkpoint_compat import (
    UnsupportedMLXCheckpointError,
    discover_mlx_checkpoint,
    is_mlx_dit_checkpoint,
    mlx_checkpoint_missing_hint,
    nvidia_fastwan_qad_reason,
    raise_if_unsupported_mlx_checkpoint,
    resolve_mlx_checkpoint,
)


def _write(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_packed_mlx_dit_is_detected(tmp_path: Path) -> None:
    _write(tmp_path / "mlx_dit.json")
    _write(tmp_path / "mlx_dit.safetensors", "x")
    assert is_mlx_dit_checkpoint(tmp_path)
    assert discover_mlx_checkpoint(tmp_path / "missing", tmp_path) == tmp_path
    assert resolve_mlx_checkpoint(None, tmp_path) == tmp_path
    raise_if_unsupported_mlx_checkpoint(tmp_path)


def test_explicit_mlx_checkpoint_wins_over_search_root(tmp_path: Path) -> None:
    packed = tmp_path / "packed"
    _write(packed / "mlx_dit.json")
    _write(packed / "mlx_dit.safetensors", "x")
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    assert resolve_mlx_checkpoint(explicit, packed) == explicit


def test_fastwan_qad_1_3b_directory_is_rejected(tmp_path: Path) -> None:
    nvidia = tmp_path / "FastWan-QAD-1.3B"
    nvidia.mkdir()
    with pytest.raises(UnsupportedMLXCheckpointError, match="FastMetal-QAD"):
        raise_if_unsupported_mlx_checkpoint(nvidia)
    assert nvidia_fastwan_qad_reason(nvidia) is not None


def test_fastwan_qad_fp8_hf_cache_path_is_rejected(tmp_path: Path) -> None:
    nvidia = tmp_path / "models--FastVideo--FastWan-QAD-FP8-1.3B" / "snapshots" / "abc"
    nvidia.mkdir(parents=True)
    with pytest.raises(UnsupportedMLXCheckpointError, match="FastWan-QAD-FP8"):
        raise_if_unsupported_mlx_checkpoint(nvidia)


def test_cuda_overlay_layout_is_rejected_even_without_qad_name(tmp_path: Path) -> None:
    root = tmp_path / "some-local-qad"
    (root / "generator_inference_transformer").mkdir(parents=True)
    with pytest.raises(UnsupportedMLXCheckpointError, match="generator_inference_transformer"):
        raise_if_unsupported_mlx_checkpoint(root)


def test_nvfp4_quantization_config_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "mystery-wan"
    _write(
        root / "transformer" / "config.json",
        json.dumps({"quantization_config": {"quant_method": "nvfp4_qat"}}),
    )
    with pytest.raises(UnsupportedMLXCheckpointError, match="nvfp4"):
        raise_if_unsupported_mlx_checkpoint(root)


def test_legacy_int8_name_with_packed_mlx_dit_is_allowed(tmp_path: Path) -> None:
    apple = tmp_path / "FastWan-QAD-INT8-1.3B"
    _write(apple / "mlx_dit.json")
    _write(apple / "mlx_dit.safetensors", "x")
    raise_if_unsupported_mlx_checkpoint(apple)
    assert nvidia_fastwan_qad_reason(apple) is None


def test_plain_diffusers_wan_is_not_rejected(tmp_path: Path) -> None:
    root = tmp_path / "FastWan2.1-T2V-1.3B-Diffusers"
    _write(root / "transformer" / "config.json", json.dumps({"num_layers": 2}))
    raise_if_unsupported_mlx_checkpoint(root)
    assert nvidia_fastwan_qad_reason(root) is None


def test_missing_mlx_dit_hint_mentions_fastmetal(tmp_path: Path) -> None:
    nvidia = tmp_path / "FastWan-QAD-1.3B"
    nvidia.mkdir()
    hint = mlx_checkpoint_missing_hint(nvidia)
    assert "Not an MLX DiT checkpoint directory" in hint
    assert "FastMetal-1.3B-QAD" in hint
    assert "FastWan-QAD-FP8-1.3B" in hint
