# SPDX-License-Identifier: Apache-2.0
"""Pooled key reductions retain the complete padded-sequence mean."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton reductions")


@pytest.mark.parametrize("tiles,heads,scale", [(6, 2, 0.01), (6, 7, 10.0), (254, 4, 1.0), (256, 4, 0.1)])
def test_pooled_key_mean_matches_float64_reference_within_bf16_rounding(tiles, heads, scale):
    from fastvideo.attention.backends.video_sparse_attn_h3 import _pool_tiles
    from fastvideo.models.dits.minimax_h3_fusions.sp_key_mean import key_mean_from_tiles

    torch.manual_seed(511)
    sequence = tiles * 64
    key = (torch.randn((1, sequence, heads, 128), device="cuda") * scale).to(torch.bfloat16)
    counts = torch.randint(0, 65, (tiles,), device="cuda", dtype=torch.int32)
    counts[0] = 0
    # Irregular holes exercise partial tiles, not only power-of-two counts.
    positions = torch.arange(64, device="cuda")[None, :]
    keep = positions < counts[:, None]
    key.view(1, tiles, 64, heads, 128).masked_fill_(~keep[None, :, :, None, None], 0)
    with torch.inference_mode():
        pool = _pool_tiles(key, counts, 64).float()
        actual = key_mean_from_tiles(pool, counts, sequence)
        reference = key.double().mean(dim=1, keepdim=True)
    # The final BF16 rounding dominates FP32 tile-reduction error. Permit one
    # BF16 spacing plus a small cancellation allowance, without requiring the
    # different FP32 reduction trees to select identical final BF16 values.
    error = (actual.double() - reference).abs()
    assert (error <= reference.abs() * 0.008 + scale * 1e-7).all()
    assert actual.shape == (1, 1, heads, 128) and actual.dtype == torch.bfloat16
