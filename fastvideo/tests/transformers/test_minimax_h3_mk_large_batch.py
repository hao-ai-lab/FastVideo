# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from fastvideo.models.dits.minimax_h3_fusions import mk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not mk.HAVE_TRITON,
                                reason="MotionKernel H3 kernels need CUDA + Triton")


def test_qkv_prologue_copies_v_above_int32_plane():
    # B*S*H*D = 4 * 40000 * 56 * 128 = 1.14688e9 > 2**30, so ``2*plane``
    # leaves int32 even though the base offsets are already int64.
    if torch.cuda.get_device_properties(0).total_memory < 16 << 30:
        pytest.skip("the shape needs ~13 GiB of device memory")
    batch, seq, heads, head_dim, rope_dim = 4, 40000, 56, 128, 96
    torch.manual_seed(0)
    q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(seq, rope_dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(seq, rope_dim, device="cuda", dtype=torch.bfloat16)
    out = mk.qkv_prologue(q, k, v, weight, weight, cos, sin)
    torch.cuda.synchronize()
    assert torch.equal(out[2 * batch:], v)
