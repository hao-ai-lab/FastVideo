# SPDX-License-Identifier: Apache-2.0
"""Construction-time validation of the FA2-only Ring kernel."""
import pytest

import fastvideo.attention.ring_attention as ring_module
from fastvideo.platforms import AttentionBackendEnum


@pytest.mark.parametrize(
    ("ring_size", "has_fa2", "expect_error"),
    [(1, False, False), (2, False, True), (2, True, False)],
)
def test_ring_fa2_dependency(monkeypatch, ring_size, has_fa2, expect_error):
    # No real process groups or GPU kernels are needed for construction.
    monkeypatch.setattr(ring_module, "get_ring_size", lambda: ring_size)
    monkeypatch.setattr(ring_module, "get_sp_world_size", lambda: 2)
    monkeypatch.setattr(ring_module, "HAS_FLASH_ATTN_2", has_fa2)
    kwargs = dict(num_heads=4, num_kv_heads=4, softmax_scale=0.125,
                  causal=False, backend=AttentionBackendEnum.FLASH_ATTN)
    if expect_error:
        with pytest.raises(RuntimeError, match="requires FlashAttention-2") as exc:
            ring_module.RingAttention.create_if_enabled(**kwargs)
        assert "ring_size=1" in str(exc.value)
        assert "flash_attn.flash_attn_interface" in str(exc.value)
    else:
        attention = ring_module.RingAttention.create_if_enabled(**kwargs)
        assert (attention is None) == (ring_size == 1)
