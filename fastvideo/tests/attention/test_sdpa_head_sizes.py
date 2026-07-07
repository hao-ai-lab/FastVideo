# SPDX-License-Identifier: Apache-2.0
"""CPU tests: SDPA must not restrict attention head sizes.

``torch.nn.functional.scaled_dot_product_attention`` is head-size
agnostic, but ``SDPABackend.get_supported_head_sizes()`` used to declare
a list copied from FlashAttentionBackend that omitted e.g. 80 (the head
size of CLIP-L vision encoders such as matrixgame2's). That made
capability validation flag SDPA as incompatible with head-80 layers even
though it handles them fine.

CPU-only; no GPU or flash-attn dependency.
"""

import torch

from fastvideo.attention.backends.sdpa import SDPABackend, SDPAImpl, SDPAMetadata
from fastvideo.platforms.cuda import CudaPlatformBase
from fastvideo.platforms.interface import AttentionBackendEnum


def test_sdpa_head_sizes_unrestricted() -> None:
    """SDPA declares no head-size restriction (None), so 80 is allowed."""
    assert SDPABackend.get_supported_head_sizes() is None
    # Once the capability-validation API (#1494) lands, head-80 must pass.
    validate = getattr(SDPABackend, "validate_compatibility", None)
    if validate is not None:
        assert validate(head_size=80, dtype=torch.bfloat16) is None


def test_explicit_torch_sdpa_pin_head_80_resolves() -> None:
    """An explicit TORCH_SDPA pin on a head-80 layer resolves to SDPABackend."""
    cls_str = CudaPlatformBase.get_attn_backend_cls(AttentionBackendEnum.TORCH_SDPA, head_size=80, dtype=torch.bfloat16)
    assert cls_str == "fastvideo.attention.backends.sdpa.SDPABackend"


def test_sdpa_impl_computes_head_80_attention() -> None:
    """SDPAImpl produces correct output for head_size=80 (not in the old list)."""
    bs, seq, heads, head_dim = 2, 8, 2, 80
    softmax_scale = head_dim**-0.5
    impl = SDPAImpl(num_heads=heads, head_size=head_dim, causal=False, softmax_scale=softmax_scale)
    torch.manual_seed(0)
    q, k, v = (torch.randn(bs, seq, heads, head_dim) for _ in range(3))

    out = impl.forward(q, k, v, SDPAMetadata(current_timestep=0))

    assert out.shape == (bs, seq, heads, head_dim)
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    ref = torch.softmax(qt @ kt.transpose(-2, -1) * softmax_scale, dim=-1) @ vt
    torch.testing.assert_close(out, ref.transpose(1, 2), rtol=1e-4, atol=1e-4)
