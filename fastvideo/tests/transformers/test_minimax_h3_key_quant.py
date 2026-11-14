# SPDX-License-Identifier: Apache-2.0
"""SageAttention FP4 bytes and prefix FP8 bytes after fused key smoothing."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires a Blackwell GPU and the native FP4 attention extension",
)


@pytest.mark.parametrize("batch,tokens,heads", [(1, 384, 56), (2, 256, 3)])
def test_smoothed_key_packing_matches_separate_sage_and_prefix_preparation(batch, tokens, heads):
    from fastvideo.models.dits.minimax_h3_fusions.key_quant import quantize_smoothed_keys
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _api, _fp8_per_token

    torch.manual_seed(1362)
    key, value = (torch.randn((batch, tokens, heads, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    key[:, 0] = 0
    index = torch.randperm(tokens, device="cuda")[:tokens - 31]
    inverse = torch.full((tokens,), -1, device="cuda", dtype=torch.int64)
    inverse[index] = torch.arange(index.numel(), device="cuda")
    with torch.inference_mode():
        mean = key.mean(dim=1, keepdim=True)
        smooth = key - mean
        expected_codes, expected_scales = _api().scale_and_quant_fp4_permute(smooth.transpose(1, 2))
        expected_key = _fp8_per_token(smooth).index_select(1, index)
        expected_value = value.index_select(1, index)
        (codes, scales), prefix_key, prefix_value = quantize_smoothed_keys(key, value, mean, inverse, index.numel())
    assert torch.equal(codes, expected_codes)
    assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))
    assert torch.equal(prefix_key, expected_key)
    assert torch.equal(prefix_value, expected_value)
