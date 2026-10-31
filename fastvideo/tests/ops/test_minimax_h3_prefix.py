# SPDX-License-Identifier: Apache-2.0
"""Exact native prefix parity across query/key tail masks and the Omni geometry."""
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention


# Enough independent query blocks keep Torch on its original unsplit K traversal.
# Low-occupancy shapes use Torch's split-K heuristic and are deliberately not dispatched.
@pytest.mark.parametrize('shape', [(2, 7, 519, 777), (2, 56, 256, 384), (2, 56, 257, 513), (1, 56, 15360, 29678)])
@pytest.mark.parametrize('magnitude', [1.0, 8.0])
@torch.inference_mode()
def test_prefix_preserves_sdpa_output_bytes(shape, magnitude):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        pytest.skip('requires GB10')
    native = pytest.importorskip('_fastvideo_h3_prefix')
    from fastvideo.models.dits.minimax_h3_fusions.prefix_attention import try_prefix_attention

    batch, heads, nq, nk = shape
    torch.manual_seed(nq + nk)
    q = (torch.randn((batch, nq, heads, 128), device='cuda', dtype=torch.bfloat16) * magnitude).transpose(1, 2)
    k = torch.randn((batch, heads, nk, 128), device='cuda', dtype=torch.bfloat16) * magnitude
    v = torch.randn_like(k)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected = scaled_dot_product_attention(q, k, v)
    actual = native.forward(q, k, v)
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    selected = try_prefix_attention(q, k, v)
    if shape == (1, 56, 15360, 29678):
        assert selected is not None
        assert torch.equal(selected.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    else:
        assert selected is None


@pytest.mark.parametrize('shape', [(1, 4, 256, 384), (1, 7, 257, 513)])
@torch.inference_mode()
def test_low_occupancy_shapes_keep_torch_splitkv_dispatch(shape):
    from fastvideo.models.dits.minimax_h3_fusions.prefix_attention import try_prefix_attention

    batch, heads, nq, nk = shape
    q = torch.empty((batch, heads, nq, 128), dtype=torch.bfloat16)
    k = torch.empty((batch, heads, nk, 128), dtype=torch.bfloat16)
    assert try_prefix_attention(q, k, k) is None
