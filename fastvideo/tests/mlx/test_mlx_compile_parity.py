# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the MLX FastWan ``mx.compile`` path.

The dense MLX runtime may opt into ``mx.compile`` with
``FASTVIDEO_MLX_COMPILE=1``. Python scalar constants must stay on the MLX trace:
NumPy scalar arithmetic evaluates traced arrays and causes the runtime to fall
back to eager execution.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for compile-parity tests")

from fastvideo.mlx_runtime.fastwan import (  # noqa: E402
    MLXWanTransformerBlock,
    gelu_tanh,
    timestep_embedding,
)


_RNG = np.random.default_rng(0)


def _rand(*shape: int, scale: float = 1.0) -> "mx.array":
    return mx.array((_RNG.standard_normal(shape) * scale).astype(np.float32))


def test_gelu_tanh_compiles_and_matches_eager() -> None:
    x = _rand(1, 120, 64)

    eager = gelu_tanh(x)
    compiled = mx.compile(gelu_tanh)(x)
    mx.eval(eager, compiled)

    np.testing.assert_array_equal(np.array(eager), np.array(compiled))


def test_timestep_embedding_compiles_and_matches_eager() -> None:
    timestep = mx.array(np.array([0.0, 250.0, 500.0, 1000.0], dtype=np.float32))
    dim = 64

    eager = timestep_embedding(timestep, dim)
    compiled = mx.compile(lambda steps: timestep_embedding(steps, dim))(timestep)
    mx.eval(eager, compiled)

    np.testing.assert_allclose(np.array(eager), np.array(compiled), rtol=1e-4, atol=1e-4)


def _tiny_block_weights(dim: int, ffn_dim: int) -> dict[str, "mx.array"]:
    square = ("to_q", "to_k", "to_v", "to_out", "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out")
    weights = {f"{name}.weight": _rand(dim, dim, scale=0.05) for name in square}
    weights.update({f"{name}.bias": _rand(dim, scale=0.05) for name in ("to_q", "to_k", "to_v", "to_out")})
    weights.update({
        "scale_shift_table": _rand(1, 6, dim, scale=0.05),
        "norm_q.weight": _rand(dim, scale=0.05),
        "norm_k.weight": _rand(dim, scale=0.05),
        "self_attn_residual_norm.norm.weight": _rand(dim, scale=0.05),
        "self_attn_residual_norm.norm.bias": _rand(dim, scale=0.05),
        "attn2.norm_q.weight": _rand(dim, scale=0.05),
        "attn2.norm_k.weight": _rand(dim, scale=0.05),
        "ffn.fc_in.weight": _rand(ffn_dim, dim, scale=0.05),
        "ffn.fc_in.bias": _rand(ffn_dim, scale=0.05),
        "ffn.fc_out.weight": _rand(dim, ffn_dim, scale=0.05),
        "ffn.fc_out.bias": _rand(dim, scale=0.05),
    })
    return weights


def test_transformer_block_compiles_and_matches_eager() -> None:
    dim, num_heads, head_dim, ffn_dim, sequence, context = 64, 4, 16, 128, 120, 32
    block = MLXWanTransformerBlock(
        _tiny_block_weights(dim, ffn_dim), dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, eps=1e-6)
    hidden_states = _rand(1, sequence, dim, scale=0.05)
    encoder_hidden_states = _rand(1, context, dim, scale=0.05)
    temb = _rand(1, 6, dim, scale=0.05)
    cos = _rand(sequence, head_dim)
    sin = _rand(sequence, head_dim)

    eager = block(hidden_states, encoder_hidden_states, temb, (cos, sin))
    compiled = mx.compile(lambda hidden, encoder, embedding, cosine, sine: block(
        hidden, encoder, embedding, (cosine, sine)))(hidden_states, encoder_hidden_states, temb, cos, sin)
    mx.eval(eager, compiled)

    np.testing.assert_allclose(np.array(eager), np.array(compiled), rtol=1e-5, atol=1e-5)
