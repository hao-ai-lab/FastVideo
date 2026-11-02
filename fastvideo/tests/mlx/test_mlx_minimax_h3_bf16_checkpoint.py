# SPDX-License-Identifier: Apache-2.0
"""Dense BF16 conversion of the MLX H3 DiT: converter round trip and forward parity.

A tiny random diffusers-layout checkpoint is written as BF16 safetensors, run
through ``convert_minimax_h3_mlx.py --formats bf16``, reloaded, and compared
with the in-memory model built from the same source. Runs on any MLX backend.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for the BF16 checkpoint tests")
torch = pytest.importorskip("torch", reason="PyTorch supplies the tiny H3 weights")

from fastvideo.mlx_runtime.fastwan import QuantizedMatrix  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    H3_MANIFEST_FILENAME,
    VSA_GATE_KEY_SUFFIX,
    load_mlx_h3_checkpoint,
    mlx_h3_checkpoint_vsa_capable,
    mlx_h3_dit_from_diffusers_safetensors,
)
from fastvideo.tests.mlx.tiny_h3 import (  # noqa: E402
    TINY_ARCH,
    build_hf_config,
    build_inputs,
    build_tiny_h3_config,
    build_torch_model,
    torch_key_to_mlx,
)

_SCRIPT = (Path(__file__).resolve().parents[3] / "scripts" / "checkpoint_conversion" / "convert_minimax_h3_mlx.py")


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_minimax_h3_mlx", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bf16_source(root: Path, *, with_gates: bool) -> None:
    model = build_torch_model()
    arrays = {}
    for torch_name, param in model.state_dict().items():
        name = torch_key_to_mlx(torch_name)
        if name.startswith("rope."):
            continue
        arrays[name] = mx.array(param.detach().float().numpy()).astype(mx.bfloat16)
    if with_gates:
        inner = TINY_ARCH["num_attention_heads"] * TINY_ARCH["attention_head_dim"]
        rng = np.random.default_rng(7)
        for index in range(TINY_ARCH["num_layers"]):
            gate = rng.standard_normal((inner, TINY_ARCH["hidden_size"])).astype(np.float32) * 0.02
            arrays[f"transformer_blocks.{index}.{VSA_GATE_KEY_SUFFIX}"] = mx.array(gate).astype(mx.bfloat16)
    root.mkdir(parents=True)
    mx.save_safetensors(str(root / "diffusion_pytorch_model.safetensors"), arrays)
    (root / "config.json").write_text(json.dumps(build_hf_config(build_tiny_h3_config())))


def _cache_timesteps(mlx_inputs) -> np.ndarray:
    return np.unique(np.concatenate([mlx_inputs["timesteps"], np.array([1.0], dtype=np.float32)]))


def _convert(tmp_path: Path, monkeypatch, timesteps: np.ndarray, *extra: str) -> Path:
    converter = _load_converter()
    monkeypatch.setattr(converter, "_adaln_cache_timesteps", lambda model_root=None: timesteps)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "convert", "--model-root", str(tmp_path / "src"), "--out", str(out), "--formats", "bf16", *extra
    ])
    converter.main()
    return out / "bf16"


def _cached_forward(dit, layout, mlx_inputs):
    video, audio = dit.forward_with_cache(
        mx.array(mlx_inputs["video_rows"]),
        mx.array(mlx_inputs["audio_rows"]),
        mx.array(mlx_inputs["text_rows"]),
        layout=layout,
        step_timesteps=mlx_inputs["timesteps"],
        row_timestep_inverse=mlx_inputs["timestep_indices"],
    )
    mx.eval(video, audio)
    return np.asarray(video.astype(mx.float32)), np.asarray(audio.astype(mx.float32))


def test_bf16_format_is_accepted_but_not_default() -> None:
    converter = _load_converter()
    assert "bf16" in converter.SUPPORTED_FORMATS
    assert "bf16" not in converter.DEFAULT_FORMATS.split()


def test_bf16_conversion_round_trip_matches_in_memory_model(tmp_path, monkeypatch, distributed_setup) -> None:
    _write_bf16_source(tmp_path / "src", with_gates=False)
    layout, _, mlx_inputs = build_inputs()
    timesteps = _cache_timesteps(mlx_inputs)

    out_dir = _convert(tmp_path, monkeypatch, timesteps)
    manifest = json.loads((out_dir / H3_MANIFEST_FILENAME).read_text())
    assert manifest["quantization"] is None
    assert manifest["quantized_keys"] == {}
    assert not mlx_h3_checkpoint_vsa_capable(out_dir)

    in_memory = mlx_h3_dit_from_diffusers_safetensors(
        tmp_path / "src", dtype="bf16", adaln_cache_timesteps=timesteps)
    loaded = load_mlx_h3_checkpoint(out_dir)

    assert loaded._adaln_cache is not None
    for block in loaded.blocks:
        assert not any(isinstance(value, QuantizedMatrix) for value in block.values())
        assert "adaln_proj.linear.weight" not in block
        assert block["attn.to_q.weight"].dtype == mx.bfloat16
        assert VSA_GATE_KEY_SUFFIX not in block
    expected = _cached_forward(in_memory, layout, mlx_inputs)
    actual = _cached_forward(loaded, layout, mlx_inputs)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert np.isfinite(actual[0]).all() and np.isfinite(actual[1]).all()


def test_bf16_conversion_keeps_dense_vsa_gates(tmp_path, monkeypatch, distributed_setup) -> None:
    _write_bf16_source(tmp_path / "src", with_gates=True)
    _, _, mlx_inputs = build_inputs()

    out_dir = _convert(tmp_path, monkeypatch, _cache_timesteps(mlx_inputs), "--include-vsa")
    assert mlx_h3_checkpoint_vsa_capable(out_dir)
    loaded = load_mlx_h3_checkpoint(out_dir)
    assert loaded.vsa_capable
    for block in loaded.blocks:
        gate = block["attn.to_gate_compress.weight"]
        assert not isinstance(gate, QuantizedMatrix)
        assert gate.dtype == mx.bfloat16
