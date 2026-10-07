# SPDX-License-Identifier: Apache-2.0
"""Check the pruned model's shared rank-16 modulation against NumPy math."""
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from fastvideo.mlx_runtime.fastwan import timestep_embedding
from fastvideo.mlx_runtime.minimax_h3 import (
    MLXMiniMaxH3DiT,
    MiniMaxH3SchedulerState,
    adaln_timestep_union,
    load_mlx_h3_checkpoint,
    save_mlx_h3_checkpoint,
)


def test_rank16_cache_has_one_silu_before_shared_basis(tmp_path):
    rng = np.random.default_rng(2026)
    hidden, rank = 32, 16

    def array(shape):
        return mx.array(rng.normal(0, 0.1, shape).astype(np.float32))

    config = dict(hidden_size=hidden, num_attention_heads=1, attention_head_dim=hidden, ffn_dim=64,
                  in_channels=24, audio_in_channels=24, patch_size=[1, 1, 1], text_dim=hidden,
                  freq_dim=hidden, time_embed_dim=hidden, rope_freq_dim=4, rope_theta=10000.,
                  norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5, adaln_rank=rank, num_layers=42)
    weights = {
        'time_embedder.linear_1.weight': array((hidden, hidden)),
        'time_embedder.linear_1.bias': array((hidden,)),
        'time_embedder.linear_2.weight': array((hidden, hidden)),
        'time_embedder.linear_2.bias': array((hidden,)),
        'adaln_basis.weight': array((rank, hidden)),
        'norm_out.linear.weight': array((2 * hidden, rank)),
        'norm_out.linear.bias': array((2 * hidden,)),
    }
    blocks = [{'attn.to_q.weight': array((hidden, hidden)),
               'adaln_proj.linear.weight': array((18 * hidden, rank)),
               'adaln_proj.linear.bias': array((18 * hidden,))} for _ in range(42)]
    dit = MLXMiniMaxH3DiT(weights, blocks, [], config)
    rungs = [999, 874, 749, 624, 500, 375, 250, 125]
    timesteps = adaln_timestep_union(MiniMaxH3SchedulerState.from_dmd_steps(10, rungs),
                                   MiniMaxH3SchedulerState.from_dmd_steps(3, rungs))

    def project(x, weight, bias=None):
        result = x @ np.array(weight).T
        return result if bias is None else result + np.array(bias)

    def silu(x):
        return x / (1 + np.exp(-x))

    features = np.array(timestep_embedding(mx.array(timesteps), hidden))
    first = project(features, weights['time_embedder.linear_1.weight'], weights['time_embedder.linear_1.bias'])
    second = project(silu(first), weights['time_embedder.linear_2.weight'], weights['time_embedder.linear_2.bias'])
    expected_basis = project(silu(second), weights['adaln_basis.weight'])
    np.testing.assert_allclose(np.array(dit.compute_temb(mx.array(timesteps))), expected_basis, atol=1e-6)
    expected_blocks = [project(expected_basis, block['adaln_proj.linear.weight'],
                               block['adaln_proj.linear.bias']).reshape(-1, 6 * hidden) for block in blocks]
    expected_out = project(expected_basis, weights['norm_out.linear.weight'], weights['norm_out.linear.bias'])
    cache = dit.precompute_adaln(timesteps)
    for tables, expected in zip(cache.block_tables, expected_blocks, strict=True):
        np.testing.assert_allclose(np.concatenate([np.array(t) for t in tables], axis=-1), expected, atol=1e-6)
    np.testing.assert_allclose(np.array(cache.norm_out_shift), expected_out[:, :hidden], atol=1e-6)
    np.testing.assert_allclose(np.array(cache.norm_out_scale), expected_out[:, hidden:], atol=1e-6)
    assert all(block['adaln_proj.linear.weight'] is None for block in blocks)

    save_mlx_h3_checkpoint(dit, tmp_path)
    loaded = load_mlx_h3_checkpoint(tmp_path)
    assert loaded.adaln_rank == rank
    assert len(loaded.blocks) == 42
    np.testing.assert_array_equal(loaded._adaln_cache.timesteps, timesteps)
    np.testing.assert_array_equal(np.array(loaded._adaln_cache.norm_out_scale), np.array(cache.norm_out_scale))
