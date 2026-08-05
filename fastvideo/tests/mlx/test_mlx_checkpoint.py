# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for pre-quantized MLX DiT checkpoints.

Saving an already-quantized DiT and reloading it must reproduce the original
forward pass exactly (same packed weights, scales, and config), for both plain
fp32 and int8-quantized models. Runs on any MLX backend (Metal or CPU).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for checkpoint tests")

from fastvideo.mlx_runtime.checkpoint import (  # noqa: E402
    MANIFEST_FILENAME,
    load_mlx_dit_checkpoint,
    save_mlx_dit_checkpoint,
)
from fastvideo.mlx_runtime.fastwan import (  # noqa: E402
    MLXQuantizationSpec,
    QuantizedMatrix,
)
from fastvideo.tests.mlx.tiny_wan import (  # noqa: E402
    build_hf_config,
    build_inputs,
    build_tiny_wan_config,
    build_torch_model,
    mlx_dit_from_torch_model,
    mlx_output,
    mlx_rotary_embeddings,
)


def _forward(dit) -> np.ndarray:
    hidden_states, encoder_hidden_states, timestep = build_inputs()
    return mlx_output(dit, hidden_states, encoder_hidden_states, timestep, mlx_rotary_embeddings(hidden_states))


def test_plain_checkpoint_round_trip(distributed_setup, tmp_path) -> None:
    dit = mlx_dit_from_torch_model(build_torch_model(), build_hf_config(build_tiny_wan_config()))
    before = _forward(dit)

    save_mlx_dit_checkpoint(dit, tmp_path / "ckpt")
    loaded = load_mlx_dit_checkpoint(tmp_path / "ckpt")

    # JSON canonicalizes tuples to lists (matching what the Diffusers
    # config.json loader produces), so compare JSON-normalized configs.
    assert json.loads(json.dumps(loaded.config)) == json.loads(json.dumps(dit.config))
    assert len(loaded.blocks) == len(dit.blocks)
    after = _forward(loaded)
    np.testing.assert_array_equal(before, after)


def test_int8_checkpoint_round_trip_skips_requantization(distributed_setup, tmp_path) -> None:
    spec = MLXQuantizationSpec.from_name("int8")
    dit = mlx_dit_from_torch_model(build_torch_model(), build_hf_config(build_tiny_wan_config()),
                                   quantization=spec)
    before = _forward(dit)

    save_mlx_dit_checkpoint(dit, tmp_path / "ckpt-int8")
    loaded = load_mlx_dit_checkpoint(tmp_path / "ckpt-int8")

    # The loaded model must carry the same packed quantized weights -- no
    # requantization pass -- and produce a bitwise-identical forward.
    reloaded_q = loaded.weights["proj_out.weight"]
    original_q = dit.weights["proj_out.weight"]
    assert isinstance(reloaded_q, QuantizedMatrix)
    assert reloaded_q.spec == spec
    np.testing.assert_array_equal(np.array(reloaded_q.weight), np.array(original_q.weight))
    np.testing.assert_array_equal(np.array(reloaded_q.scales), np.array(original_q.scales))

    block_q = loaded.blocks[0].weights["to_q.weight"]
    assert isinstance(block_q, QuantizedMatrix)

    after = _forward(loaded)
    np.testing.assert_array_equal(before, after)


def test_missing_checkpoint_raises_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Not an MLX DiT checkpoint"):
        load_mlx_dit_checkpoint(tmp_path / "does-not-exist")


def test_future_format_version_is_rejected(distributed_setup, tmp_path) -> None:
    dit = mlx_dit_from_torch_model(build_torch_model(), build_hf_config(build_tiny_wan_config()))
    ckpt = save_mlx_dit_checkpoint(dit, tmp_path / "ckpt")

    manifest_path = ckpt / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 99
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="format_version=99"):
        load_mlx_dit_checkpoint(ckpt)
