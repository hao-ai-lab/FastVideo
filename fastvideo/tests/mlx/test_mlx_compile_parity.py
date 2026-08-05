# SPDX-License-Identifier: Apache-2.0
"""Guard the ``mx.compile`` path of the MLX FastWan forward.

`mx.compile` on the DiT forward was silently broken: a NumPy scalar multiplying
a traced array (``np.sqrt(2/pi) * x`` in ``gelu_tanh``) dispatched through NumPy,
which evals the traced array — illegal during compile. It raised "Attempting to
eval an array during function transformations" (caught → eager fallback, no
speedup) or segfaulted the process. These tests keep the compiled forward tracing
cleanly and numerically identical to eager, so the ~1.4x compile speedup cannot
silently regress. See ``fastvideo/mlx_runtime/fastwan.py`` and
``docs/design/apple_silicon_benchmark_baseline.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for the compile-parity tests")

from fastvideo.mlx_runtime.fastwan import (  # noqa: E402
    MLXWanTransformerBlock,
    gelu_tanh,
    timestep_embedding,
)

# One module-level generator so successive _rand calls of the same shape get
# distinct values (re-seeding per call made every (dim, dim) weight identical).
_RNG = np.random.default_rng(0)


def _rand(*shape: int, scale: float = 1.0) -> "mx.array":
    return mx.array((_RNG.standard_normal(shape) * scale).astype(np.float32))


def test_gelu_tanh_compiles_and_matches_eager() -> None:
    """The tanh-GELU (the op that broke compile) traces and matches eager."""
    x = _rand(1, 120, 64)
    eager = gelu_tanh(x)
    mx.eval(eager)
    compiled = mx.compile(gelu_tanh)(x)
    mx.eval(compiled)
    # Bit-identical on the same device is the compile-parity contract (not allclose).
    np.testing.assert_array_equal(np.array(eager), np.array(compiled))


def test_timestep_embedding_compiles_and_matches_eager() -> None:
    """``timestep_embedding`` (math.log fix sibling of gelu_tanh) compiles cleanly.

    Unlike gelu_tanh, the exp/sin/cos chain can reassociate under compile, so
    we allow a tight tolerance rather than bit-identity.
    """
    t = mx.array(np.array([0.0, 250.0, 500.0, 1000.0], dtype=np.float32))
    dim = 64
    eager = timestep_embedding(t, dim)
    mx.eval(eager)
    compiled = mx.compile(lambda steps: timestep_embedding(steps, dim))(t)
    mx.eval(compiled)
    np.testing.assert_allclose(np.array(eager), np.array(compiled), rtol=1e-4, atol=1e-4)


def _tiny_block_weights(dim: int, ffn: int) -> dict:
    square = ["to_q", "to_k", "to_v", "to_out", "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out"]
    weights = {f"{k}.weight": _rand(dim, dim, scale=0.05) for k in square}
    weights.update({f"{k}.bias": _rand(dim, scale=0.05) for k in ["to_q", "to_k", "to_v", "to_out"]})
    weights.update({
        "scale_shift_table": _rand(1, 6, dim, scale=0.05),
        "norm_q.weight": _rand(dim, scale=0.05),
        "norm_k.weight": _rand(dim, scale=0.05),
        "self_attn_residual_norm.norm.weight": _rand(dim, scale=0.05),
        "self_attn_residual_norm.norm.bias": _rand(dim, scale=0.05),
        "attn2.norm_q.weight": _rand(dim, scale=0.05),
        "attn2.norm_k.weight": _rand(dim, scale=0.05),
        "ffn.fc_in.weight": _rand(ffn, dim, scale=0.05),
        "ffn.fc_in.bias": _rand(ffn, scale=0.05),
        "ffn.fc_out.weight": _rand(dim, ffn, scale=0.05),
        "ffn.fc_out.bias": _rand(dim, scale=0.05),
    })
    return weights


def test_transformer_block_compiles_and_matches_eager() -> None:
    """The full dense block (the mx.compile target's body) traces and matches."""
    dim, num_heads, head_dim, ffn, seq, ctx = 64, 4, 16, 128, 120, 32
    block = MLXWanTransformerBlock(_tiny_block_weights(dim, ffn), dim=dim, ffn_dim=ffn, num_heads=num_heads, eps=1e-6)
    hidden = _rand(1, seq, dim, scale=0.05)
    context = _rand(1, ctx, dim, scale=0.05)
    temb = _rand(1, 6, dim, scale=0.05)
    cos = _rand(seq, head_dim)
    sin = _rand(seq, head_dim)

    eager = block(hidden, context, temb, (cos, sin))
    mx.eval(eager)
    compiled = mx.compile(lambda h, e, t, c, s: block(h, e, t, (c, s)))(hidden, context, temb, cos, sin)
    mx.eval(compiled)

    np.testing.assert_array_equal(np.array(eager), np.array(compiled))
