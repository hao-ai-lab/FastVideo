# SPDX-License-Identifier: Apache-2.0
"""sm89 INT8-QK/FP8-PV regression against dense masked BF16 attention."""
from __future__ import annotations

import pytest
import torch


def _cuda_sm89():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("RTX 4090 / sm89 CUDA is required")


@pytest.mark.parametrize("partial", [False, True])
def test_sparse_int8_preserves_tile_selection_and_valid_keys(partial):
    _cuda_sm89()
    from fastvideo.attention.backends.minimax_h3_sparse_int8 import sparse_sm89_attention

    torch.manual_seed(42)
    q, k, v = (torch.randn(1, 2, 256, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    vbs = torch.tensor([64, 7 if partial else 64, 31 if partial else 64, 64], device="cuda", dtype=torch.int32)
    valid = torch.arange(256, device="cuda") % 64 < vbs.repeat_interleave(64)
    k[..., ~valid, :] = 0
    v[..., ~valid, :] = 0
    # Adjacent query tiles deliberately select different keys. A paired-query
    # OR adapter would fail this regression even with perfect quantization.
    mask = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]],
                         device="cuda", dtype=torch.bool)[None, None].expand(1, 2, -1, -1).contiguous()
    dense_mask = mask.repeat_interleave(64, -2).repeat_interleave(64, -1) & valid[None, None, None, :]
    with torch.inference_mode():
        expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                                    attn_mask=dense_mask)
        output = sparse_sm89_attention(q, k, v, mask, vbs)
    assert torch.isfinite(output).all()
    relative_error = (output.float() - expected).norm() / expected.norm()
    assert relative_error < 0.055, float(relative_error)
    torch.testing.assert_close(output.float(), expected, atol=0.05, rtol=0.15)


def test_sparse_int8_handles_empty_selection():
    _cuda_sm89()
    from fastvideo.attention.backends.minimax_h3_sparse_int8 import sparse_sm89_attention

    q = torch.zeros(1, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        out = sparse_sm89_attention(q, q, q, torch.zeros(1, 1, 2, 2, device="cuda", dtype=torch.bool),
                                        torch.tensor([64, 64], device="cuda", dtype=torch.int32))
    assert torch.count_nonzero(out) == 0


def test_fp8_dynamic_probability_scale_preserves_small_blocks():
    """A large earlier max must not erase a later block's small P but large V."""
    _cuda_sm89()
    from fastvideo.attention.backends.minimax_h3_sparse_int8 import sparse_sm89_attention

    q = torch.zeros(1, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    q[..., 0] = 16
    k[..., :64, 0] = 14
    k[..., 64:, 0] = 2
    v[..., 64:, :] = 1e7
    mask = torch.ones(1, 1, 2, 2, device="cuda", dtype=torch.bool)
    vbs = torch.tensor([64, 64], device="cuda", dtype=torch.int32)
    with torch.inference_mode():
        reference = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        fixed = sparse_sm89_attention(q, k, v, mask, vbs, int8_qk=False, fp8_pv=True,
                                       fp8_v_tiles=True, fp8_dynamic_p=False)
        dynamic = sparse_sm89_attention(q, k, v, mask, vbs, int8_qk=False, fp8_pv=True,
                                         fp8_v_tiles=True, fp8_dynamic_p=True)
    assert reference.abs().min() > 0.1
    assert torch.count_nonzero(fixed) == 0
    torch.testing.assert_close(dynamic.float(), reference, rtol=0.02, atol=0.02)
