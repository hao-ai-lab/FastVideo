# SPDX-License-Identifier: Apache-2.0
"""MLXWanPipeline.__init__ validation.

The constructor only does filesystem/path checks (no real weights loaded, no
mlx.core import) -- these run for real here, no stubbing, since nothing in
__init__ needs Apple Silicon. Generation itself (.generate()) does need
Metal and is out of scope for this file; see mlx_wan_server tests for the
serving-layer coverage that stubs it out.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastvideo.mlx_runtime.wan_pipeline import MLXWan22Pipeline, MLXWanPipeline


def _make_model_root(tmp_path: Path) -> Path:
    """A model root with just enough structure to pass the constructor's checks."""
    model_root = tmp_path / "model_root"
    (model_root / "tokenizer").mkdir(parents=True)
    (model_root / "text_encoder").mkdir(parents=True)
    return model_root


def _make_packed_checkpoint(tmp_path: Path, name: str = "FastMetal-1.3B-QAD-mlx", in_channels: int | None = None) -> Path:
    """A directory shaped like a real packed MLX DiT checkpoint."""
    checkpoint = tmp_path / name
    checkpoint.mkdir(parents=True)
    manifest = {"config": {"in_channels": in_channels}} if in_channels is not None else {}
    (checkpoint / "mlx_dit.json").write_text(json.dumps(manifest))
    (checkpoint / "mlx_dit.safetensors").write_bytes(b"")
    return checkpoint


def test_init_accepts_a_valid_model_root_and_checkpoint(tmp_path) -> None:
    model_root = _make_model_root(tmp_path)
    checkpoint = _make_packed_checkpoint(tmp_path)
    pipeline = MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)
    assert pipeline.model_root == model_root
    assert pipeline.mlx_checkpoint == checkpoint


def test_init_rejects_missing_tokenizer_dir(tmp_path) -> None:
    model_root = tmp_path / "model_root"
    (model_root / "text_encoder").mkdir(parents=True)
    checkpoint = _make_packed_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


def test_init_rejects_missing_text_encoder_dir(tmp_path) -> None:
    model_root = tmp_path / "model_root"
    (model_root / "tokenizer").mkdir(parents=True)
    checkpoint = _make_packed_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="text_encoder"):
        MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


def test_init_rejects_nvidia_fastwan_qad_checkpoint_name(tmp_path) -> None:
    """FastWan-QAD is the NVIDIA NVFP4/FP8 release; loading it through MLX
    silently requantizes the wrong weights (see checkpoint_compat.py)."""
    model_root = _make_model_root(tmp_path)
    nvidia_checkpoint = tmp_path / "FastWan-QAD-1.3B"
    nvidia_checkpoint.mkdir()
    with pytest.raises(ValueError, match="FastWan-QAD"):
        MLXWanPipeline(model_root=model_root, mlx_checkpoint=nvidia_checkpoint)


def test_init_allows_the_legacy_int8_nvidia_named_directory(tmp_path) -> None:
    """FastWan-QAD-INT8-* predates the mlx_dit.json packing convention but is
    a real Apple checkpoint; the name-based NVIDIA check must not reject it."""
    model_root = _make_model_root(tmp_path)
    checkpoint = _make_packed_checkpoint(tmp_path, name="FastWan-QAD-INT8-1.3B")
    MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


def test_init_does_not_validate_checkpoint_contents(tmp_path) -> None:
    """A directory that is neither a packed MLX checkpoint nor NVIDIA-flagged
    passes __init__ unexamined -- real content validation happens inside
    generate() when the weights are actually loaded (needs Metal)."""
    model_root = _make_model_root(tmp_path)
    empty_checkpoint = tmp_path / "not_a_real_checkpoint"
    empty_checkpoint.mkdir()
    MLXWanPipeline(model_root=model_root, mlx_checkpoint=empty_checkpoint)


def test_init_rejects_a_wan22_checkpoint(tmp_path) -> None:
    """Pointing the Wan2.1 pipeline at a 48-channel FastMetal-5B-QAD checkpoint
    would silently produce garbled output (wrong VAE compression assumed) --
    this must be caught here, not discovered downstream."""
    model_root = _make_model_root(tmp_path)
    wan22_checkpoint = _make_packed_checkpoint(tmp_path, name="FastMetal-5B-QAD", in_channels=48)
    with pytest.raises(ValueError, match="Wan2.2-TI2V"):
        MLXWanPipeline(model_root=model_root, mlx_checkpoint=wan22_checkpoint)


def test_init_accepts_a_checkpoint_with_declared_16_channels(tmp_path) -> None:
    model_root = _make_model_root(tmp_path)
    checkpoint = _make_packed_checkpoint(tmp_path, in_channels=16)
    MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


class TestMLXWan22Pipeline:
    """Mirrors the MLXWanPipeline coverage above for the Wan2.2-TI2V (5B) pipeline."""

    def test_init_accepts_a_valid_model_root_and_checkpoint(self, tmp_path) -> None:
        model_root = _make_model_root(tmp_path)
        checkpoint = _make_packed_checkpoint(tmp_path, name="FastMetal-5B-QAD", in_channels=48)
        pipeline = MLXWan22Pipeline(model_root=model_root, mlx_checkpoint=checkpoint)
        assert pipeline.model_root == model_root

    def test_init_rejects_missing_tokenizer_dir(self, tmp_path) -> None:
        model_root = tmp_path / "model_root"
        (model_root / "text_encoder").mkdir(parents=True)
        checkpoint = _make_packed_checkpoint(tmp_path, name="FastMetal-5B-QAD", in_channels=48)
        with pytest.raises(FileNotFoundError, match="tokenizer"):
            MLXWan22Pipeline(model_root=model_root, mlx_checkpoint=checkpoint)

    def test_init_rejects_a_wan21_checkpoint(self, tmp_path) -> None:
        """The reverse mistake: pointing the 5B pipeline at a 1.3B/14B checkpoint."""
        model_root = _make_model_root(tmp_path)
        wan21_checkpoint = _make_packed_checkpoint(tmp_path, in_channels=16)
        with pytest.raises(ValueError, match="MLXWanPipeline for 1.3B/14B"):
            MLXWan22Pipeline(model_root=model_root, mlx_checkpoint=wan21_checkpoint)

    def test_init_does_not_validate_checkpoint_contents(self, tmp_path) -> None:
        model_root = _make_model_root(tmp_path)
        empty_checkpoint = tmp_path / "not_a_real_checkpoint"
        empty_checkpoint.mkdir()
        MLXWan22Pipeline(model_root=model_root, mlx_checkpoint=empty_checkpoint)
