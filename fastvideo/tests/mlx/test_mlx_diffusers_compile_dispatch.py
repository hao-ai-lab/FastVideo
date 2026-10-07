# SPDX-License-Identifier: Apache-2.0
"""Regression tests for preserving MLX compile flags on raw Diffusers loads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX required for compile dispatch")


def _write_tiny_diffusers_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    import torch
    from safetensors.torch import save_file

    config = {
        "num_layers": 0,
        "num_attention_heads": 1,
        "attention_head_dim": 2,
        "ffn_dim": 4,
        "in_channels": 16,
        "out_channels": 16,
        "patch_size": [1, 2, 2],
        "freq_dim": 2,
        "eps": 1e-6,
    }
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "diffusion_pytorch_model.safetensors"
    config_path.write_text(json.dumps(config))

    hidden = 2
    weights = {
        "patch_embedding.weight": torch.zeros(hidden, 64),
        "condition_embedder.time_embedder.linear_1.weight": torch.zeros(hidden, 2),
        "condition_embedder.time_embedder.linear_2.weight": torch.zeros(hidden, hidden),
        "condition_embedder.time_proj.weight": torch.zeros(hidden * 6, hidden),
        "condition_embedder.text_embedder.linear_1.weight": torch.zeros(hidden, 4096),
        "condition_embedder.text_embedder.linear_2.weight": torch.zeros(hidden, hidden),
        "scale_shift_table": torch.zeros(2, hidden),
        "proj_out.weight": torch.zeros(64, hidden),
    }
    save_file(weights, checkpoint_path)
    return checkpoint_path, config_path


def test_diffusers_loader_preserves_compile_flag(tmp_path: Path) -> None:
    from fastvideo.mlx_runtime.fastwan import mlx_dit_from_diffusers_safetensors

    checkpoint_path, config_path = _write_tiny_diffusers_checkpoint(tmp_path)

    compiled = mlx_dit_from_diffusers_safetensors(
        checkpoint_path,
        config_path,
        compile=True,
        num_blocks=0,
        quantization=None,
    )
    eager = mlx_dit_from_diffusers_safetensors(
        checkpoint_path,
        config_path,
        compile=False,
        num_blocks=0,
        quantization=None,
    )

    assert compiled._enable_compile is True
    assert eager._enable_compile is False


def test_prompt_to_video_cli_passes_compile_to_diffusers_loader() -> None:
    root = Path(__file__).resolve().parents[3]
    script = root / "examples/inference/basic/mlx_wan_prompt_to_video.py"
    source = script.read_text()

    assert "compile=args.mlx_compile" in source
