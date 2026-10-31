# SPDX-License-Identifier: Apache-2.0
"""CUDA byte parity for combined FP8 prefix preparation and K/V gather."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires a Blackwell GPU",
)


@pytest.mark.parametrize("batch,heads,rows,strided", [(1, 56, 257, False), (2, 3, 19, True)])
def test_prefix_quant_gather_matches_existing_fp8_roundtrip(batch, heads, rows, strided):
    from fastvideo.models.dits.minimax_h3_fusions.prefix_quant import quantize_gather_prefix_kv
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _fp8_per_token

    torch.manual_seed(9941)
    key, value = (torch.randn((batch, rows * 2, heads, 128), device="cuda", dtype=torch.bfloat16)
                  for _ in range(2))
    if strided:
        key, value = key[:, ::2], value[:, ::2]
    key[:, 0] = 0
    key[:, 1] *= 1e-12
    index = torch.randperm(key.shape[1], device="cuda")[:rows - 1]
    with torch.inference_mode():
        expected_key = _fp8_per_token(key).index_select(1, index)
        expected_value = value.index_select(1, index)
        actual_key, actual_value = quantize_gather_prefix_kv(key, value, index)
    assert torch.equal(actual_key, expected_key)
    assert torch.equal(actual_value, expected_value)
    assert actual_key.is_contiguous() and actual_value.is_contiguous()
