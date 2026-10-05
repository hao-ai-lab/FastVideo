# SPDX-License-Identifier: Apache-2.0
"""Exercise native floating-point quantized H3 storage on supported MLX builds."""
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec, ensure_quantization_supported, linear
from fastvideo.mlx_runtime.minimax_h3 import MLXMiniMaxH3DiT, load_mlx_h3_checkpoint, quantize_matrix, save_mlx_h3_checkpoint


@pytest.mark.parametrize('mode', ['mxfp8', 'mxfp4', 'nvfp4'])
def test_float_quantized_checkpoint_preserves_matrix(tmp_path, mode):
    spec = MLXQuantizationSpec.from_name(mode)
    ensure_quantization_supported(spec)
    dense = (mx.random.normal((64, 64)) * 0.001).astype(mx.bfloat16)
    weight = quantize_matrix(dense, spec)
    restored = mx.dequantize(weight.weight, weight.scales, mode=mode).astype(mx.float32) * weight.global_scale
    relative_error = mx.sqrt(mx.sum((restored - dense.astype(mx.float32))**2) / mx.sum(dense.astype(mx.float32)**2))
    assert float(relative_error.item()) < 0.15
    x = mx.random.normal((3, 64)).astype(mx.bfloat16)
    config = dict(hidden_size=64, num_attention_heads=1, attention_head_dim=64, ffn_dim=128,
                  in_channels=24, audio_in_channels=24, patch_size=[1, 1, 1], text_dim=64,
                  freq_dim=64, time_embed_dim=64, rope_freq_dim=4, rope_theta=10000.,
                  norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5)
    dit = MLXMiniMaxH3DiT({'test.weight': weight}, [], [], config)
    save_mlx_h3_checkpoint(dit, tmp_path)
    loaded = load_mlx_h3_checkpoint(tmp_path)
    actual = linear(x, loaded.weights['test.weight']).astype(mx.float32)
    expected = linear(x, weight).astype(mx.float32)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    assert loaded.weights['test.weight'].biases is None
    assert loaded.weights['test.weight'].spec == spec
    assert loaded.weights['test.weight'].global_scale == weight.global_scale
