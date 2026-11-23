# SPDX-License-Identifier: Apache-2.0
"""Parity tests for fused and compiled MiniMax-H3 MLX video VAE decoder primitives."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from fastvideo.mlx_runtime import minimax_h3_video_vae as vae  # noqa: E402


def _as_f32(array) -> np.ndarray:
    return np.asarray(array.astype(mx.float32), dtype=np.float32)


@pytest.mark.parametrize("dtype,rtol,atol", [
    (mx.float32, 1e-5, 1e-5),
    (mx.float16, 2**-9, 2e-3),
    (mx.bfloat16, 2**-6, 1e-2),
])
def test_rms_norm_primitives_match_eager(dtype, rtol: float, atol: float) -> None:
    rng = np.random.default_rng(17)
    eps = 1e-5

    qk = mx.array(rng.standard_normal((19, 1, 4, 64)).astype(np.float32)).astype(dtype)
    qk_f32 = qk.astype(mx.float32)
    expected_qk = (qk_f32 / mx.sqrt(mx.mean(qk_f32 * qk_f32, axis=-1, keepdims=True) + eps)).astype(dtype)
    actual_qk = vae._rms_norm_no_affine(qk, eps)
    assert actual_qk.dtype == dtype
    np.testing.assert_allclose(_as_f32(actual_qk), _as_f32(expected_qk), rtol=rtol, atol=atol)

    hidden = mx.array(rng.standard_normal((19, 1, 128)).astype(np.float32)).astype(dtype)
    weight = mx.array((1.0 + rng.standard_normal((128, )) * 0.1).astype(np.float32)).astype(dtype)
    h_f32 = hidden.astype(mx.float32)
    expected_h = (h_f32 / mx.sqrt(mx.mean(h_f32 * h_f32, axis=-1, keepdims=True) + eps) * weight).astype(dtype)
    actual_h = vae._rms_norm_affine(hidden, weight, eps)
    assert actual_h.dtype == dtype
    np.testing.assert_allclose(_as_f32(actual_h), _as_f32(expected_h), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_rms_norm_affine_handles_mixed_weight_dtype(dtype) -> None:
    rng = np.random.default_rng(19)
    hidden = mx.array(rng.standard_normal((11, 1, 64)).astype(np.float32)).astype(dtype)
    weight_fp32 = mx.array((1.0 + rng.standard_normal((64, )) * 0.1).astype(np.float32))
    out = vae._rms_norm_affine(hidden, weight_fp32, 1e-5)
    assert out.dtype == dtype
    ref = mx.fast.rms_norm(hidden, weight_fp32.astype(dtype), 1e-5)
    np.testing.assert_array_equal(_as_f32(out), _as_f32(ref))


@pytest.mark.parametrize("dtype,rtol,atol", [
    (mx.float32, 1e-5, 1e-6),
    (mx.float16, 2**-10, 1e-4),
    (mx.bfloat16, 2**-7, 0.0),
])
@pytest.mark.parametrize("seq_len", [17, 29])
def test_compiled_rotary_matches_eager(dtype, rtol: float, atol: float, seq_len: int) -> None:
    rng = np.random.default_rng(23 + seq_len)
    num_heads, head_dim, rotary_dim = 4, 64, 48
    q = mx.array(rng.standard_normal((seq_len, 1, num_heads, head_dim)).astype(np.float32)).astype(dtype)
    k = mx.array(rng.standard_normal((seq_len, 1, num_heads, head_dim)).astype(np.float32)).astype(dtype)
    pos = mx.array(rng.uniform(-1.0, 1.0, size=(seq_len, 3)).astype(np.float32))
    cos, sin = vae._rotary_cos_sin(pos, rotary_dim=rotary_dim, theta=100.0)

    def eager_rotate(x):
        x_rot, x_pass = x[..., :rotary_dim].astype(mx.float32), x[..., rotary_dim:]
        first, second = mx.split(x_rot, 2, axis=-1)
        out = (x_rot * cos + mx.concatenate([-second, first], axis=-1) * sin).astype(x.dtype)
        return mx.concatenate([out, x_pass], axis=-1)

    q_actual, k_actual = vae._apply_rotary(q, k, cos, sin)
    q_expected, k_expected = eager_rotate(q), eager_rotate(k)
    assert q_actual.dtype == dtype and k_actual.dtype == dtype
    np.testing.assert_allclose(_as_f32(q_actual), _as_f32(q_expected), rtol=rtol, atol=atol)
    np.testing.assert_allclose(_as_f32(k_actual), _as_f32(k_expected), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype,rtol,atol", [
    (mx.float32, 1e-5, 1e-6),
    (mx.float16, 2**-10, 1e-4),
    (mx.bfloat16, 2**-7, 0.0),
])
@pytest.mark.parametrize("seq_len", [11, 37])
def test_compiled_swiglu_matches_eager_across_shapes(dtype, rtol: float, atol: float, seq_len: int) -> None:
    rng = np.random.default_rng(31 + seq_len)
    value = mx.array(rng.standard_normal((seq_len, 1, 256)).astype(np.float32)).astype(dtype)
    gate = mx.array(rng.standard_normal((seq_len, 1, 256)).astype(np.float32)).astype(dtype)
    actual = vae._swiglu(value, gate)
    expected = value * (gate * mx.sigmoid(gate))
    assert actual.dtype == dtype
    np.testing.assert_allclose(_as_f32(actual), _as_f32(expected), rtol=rtol, atol=atol)


def test_transformer_block_matches_eager_fp32() -> None:
    rng = np.random.default_rng(43)
    seq_len, num_heads, head_dim, ffn_mult = 21, 4, 16, 2
    dim = num_heads * head_dim
    inner = dim * ffn_mult
    eps = 1e-5

    def arr(shape, scale: float = 0.05):
        return mx.array((rng.standard_normal(shape) * scale).astype(np.float32))

    block = {
        "norm1.weight": mx.ones((dim, ), dtype=mx.float32),
        "norm2.weight": mx.ones((dim, ), dtype=mx.float32),
        "scale1": arr((dim, ), 0.1),
        "scale2": arr((dim, ), 0.1),
        "attn.to_q.weight": arr((dim, dim)),
        "attn.to_q.bias": arr((dim, )),
        "attn.to_k.weight": arr((dim, dim)),
        "attn.to_k.bias": arr((dim, )),
        "attn.to_v.weight": arr((dim, dim)),
        "attn.to_v.bias": arr((dim, )),
        "attn.to_out.0.weight": arr((dim, dim)),
        "attn.to_out.0.bias": arr((dim, )),
        "ff.net.0.proj.weight": arr((2 * inner, dim)),
        "ff.net.0.proj.bias": arr((2 * inner, )),
        "ff.net.2.weight": arr((dim, inner)),
        "ff.net.2.bias": arr((dim, )),
    }
    x = arr((seq_len, 1, dim), 1.0)
    pos = mx.array(rng.uniform(-1.0, 1.0, size=(seq_len, 3)).astype(np.float32))
    cos, sin = vae._rotary_cos_sin(pos, rotary_dim=int(head_dim * 0.75), theta=100.0)

    actual = vae._transformer_block(
        block,
        x,
        block["scale1"],
        block["scale2"],
        cos,
        sin,
        num_heads=num_heads,
        head_dim=head_dim,
        norm_eps=eps,
        qk_norm_eps=eps,
    )

    def eager_rms(t, w=None):
        norm = t / mx.sqrt(mx.mean(t * t, axis=-1, keepdims=True) + eps)
        return norm * w if w is not None else norm

    n1 = eager_rms(x, block["norm1.weight"])
    q = vae._linear(n1, block["attn.to_q.weight"], block["attn.to_q.bias"]).reshape(seq_len, 1, num_heads, head_dim)
    k = vae._linear(n1, block["attn.to_k.weight"], block["attn.to_k.bias"]).reshape(seq_len, 1, num_heads, head_dim)
    v = vae._linear(n1, block["attn.to_v.weight"], block["attn.to_v.bias"]).reshape(seq_len, 1, num_heads, head_dim)
    q, k = eager_rms(q), eager_rms(k)
    rdim = cos.shape[-1]
    q1, q2 = mx.split(q[..., :rdim], 2, axis=-1)
    k1, k2 = mx.split(k[..., :rdim], 2, axis=-1)
    q = mx.concatenate([q[..., :rdim] * cos + mx.concatenate([-q2, q1], axis=-1) * sin, q[..., rdim:]], axis=-1)
    k = mx.concatenate([k[..., :rdim] * cos + mx.concatenate([-k2, k1], axis=-1) * sin, k[..., rdim:]], axis=-1)
    attn = mx.fast.scaled_dot_product_attention(vae._ct(q, 1, 2, 0, 3),
                                                vae._ct(k, 1, 2, 0, 3),
                                                vae._ct(v, 1, 2, 0, 3),
                                                scale=head_dim**-0.5)
    attn_out = vae._linear(vae._ct(attn, 0, 2, 1, 3).reshape(seq_len, 1, dim), block["attn.to_out.0.weight"],
                           block["attn.to_out.0.bias"])
    h = x + attn_out * block["scale1"]
    n2 = eager_rms(h, block["norm2.weight"])
    ff = vae._linear(n2, block["ff.net.0.proj.weight"], block["ff.net.0.proj.bias"])
    val, gate = mx.split(ff, 2, axis=-1)
    expected = h + vae._linear(val * (gate * mx.sigmoid(gate)), block["ff.net.2.weight"],
                               block["ff.net.2.bias"]) * block["scale2"]

    np.testing.assert_allclose(_as_f32(actual), _as_f32(expected), rtol=1e-4, atol=1e-4)
