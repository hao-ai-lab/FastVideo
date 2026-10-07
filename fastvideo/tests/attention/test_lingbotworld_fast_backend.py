# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from fastvideo.models.dits.lingbotworld2 import causal_fast
from fastvideo.platforms import AttentionBackendEnum


class _SDPABackend:

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"


def test_lingbot_attention_honors_selected_sdpa_backend(monkeypatch) -> None:
    monkeypatch.setattr(causal_fast, "get_attn_backend", lambda *args, **kwargs: _SDPABackend)
    layer = causal_fast.CausalWanSelfAttention(dim=8, num_heads=2)
    assert layer.backend == AttentionBackendEnum.TORCH_SDPA

    monkeypatch.setattr(causal_fast, "flash_attention", lambda **kwargs: pytest.fail("FlashAttention was called"))
    q = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) / 24
    k = q.flip(1)
    v = q + 1

    actual = causal_fast.attention(q, k, v, layer.backend, dtype=torch.float32)
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    torch.testing.assert_close(actual, expected.transpose(1, 2).contiguous())
