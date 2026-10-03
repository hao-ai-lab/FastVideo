# SPDX-License-Identifier: Apache-2.0
"""Exercise native MXFP8 H3 checkpoint storage on supported MLX builds."""
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec, ensure_quantization_supported, linear, quantize_matrix
from fastvideo.mlx_runtime.minimax_h3 import MLXMiniMaxH3DiT, load_mlx_h3_checkpoint, save_mlx_h3_checkpoint


def test_mxfp8_checkpoint_preserves_quantized_matrix(tmp_path):
    spec = MLXQuantizationSpec.from_name('mxfp8')
    ensure_quantization_supported(spec)
    weight = quantize_matrix(mx.random.normal((64, 64)).astype(mx.bfloat16), spec)
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
