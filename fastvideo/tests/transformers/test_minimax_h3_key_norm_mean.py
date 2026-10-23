# SPDX-License-Identifier: Apache-2.0
"""Exact GB10 normalization and ATen mean parity at the specialized H3 geometry."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1),
    reason="requires the measured GB10 reduction geometry",
)


@pytest.mark.parametrize("exponent,strided", [(0, False), (6, False), (12, False), (6, True)])
def test_key_normalization_and_mean_match_original(exponent, strided):
    from fastvideo.models.dits.minimax_h3_fusions.key_norm_mean import gather_key_norm_mean
    from fastvideo.models.dits.minimax_h3_fusions.qknorm_rope import fused_qknorm_rope

    torch.manual_seed(29017 + exponent)
    rows, source_rows, heads, dim = 18432, 14859, 56, 128
    width = dim * 2 if strided else dim
    x = torch.randn((1, source_rows, heads, width), device="cuda", dtype=torch.bfloat16)
    if exponent:
        x *= torch.exp2(torch.randint(-exponent, exponent + 1, x.shape, device=x.device).to(x.dtype))
    if strided:
        x = x[..., ::2]
    index = torch.randperm(rows, device=x.device).to(torch.int64)
    index[index >= source_rows] = -1
    weight = torch.randn(dim, device=x.device, dtype=x.dtype)
    angles = torch.randn((rows, 48), device=x.device)
    angles = torch.cat((angles, angles), -1)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    with torch.inference_mode():
        expanded = x.index_select(1, index.clamp(min=0))
        expanded[:, index < 0] = 0
        expected = fused_qknorm_rope(expanded, weight, cos, sin, 1e-6)
        expected_mean = expected.mean(dim=1, keepdim=True)
        actual, actual_mean = gather_key_norm_mean(x, weight, cos, sin, 1e-6, index)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_mean, expected_mean)
    assert actual.is_contiguous() and actual_mean.is_contiguous()
