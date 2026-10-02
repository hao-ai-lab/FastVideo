# SPDX-License-Identifier: Apache-2.0
"""Weight-free regression tests for Wan's opt-in FlashInfer selection.

The API test directory is included in the unit-test CI lane. Only platform
probing and projection construction are stubbed; backend selection is real.
"""

from types import SimpleNamespace

import pytest
import torch

from fastvideo import envs, platforms
from fastvideo.attention.backends.flashinfer import FlashInferBackend, FlashInferImpl
from fastvideo.attention.selector import _cached_get_attn_backend, get_attn_backend
from fastvideo.models.wan import transformer as wan
from fastvideo.platforms.interface import AttentionBackendEnum


@pytest.mark.parametrize("prefill_backend", ["single", "cudnn"])
@pytest.mark.parametrize("cross_attention_cls", [wan.WanT2VCrossAttention, wan.WanI2VCrossAttention])
def test_wan_flashinfer_request_reaches_self_and_cross_attention(
    env_overrides, monkeypatch, prefill_backend, cross_attention_cls
):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("FLASHINFER"))
    env_overrides.enter_context(envs.FASTVIDEO_FLASHINFER_PREFILL_BACKEND.override(prefill_backend))
    selections = []

    def resolve_backend(selected_backend, head_size, dtype):
        # Before the fix, the selector changes this request to None because
        # the Wan layer does not declare support for FLASHINFER.
        assert selected_backend is AttentionBackendEnum.FLASHINFER
        assert head_size == 128
        selections.append(selected_backend)
        return "fastvideo.attention.backends.flashinfer.FlashInferBackend"

    monkeypatch.setattr(platforms, "current_platform", SimpleNamespace(get_attn_backend_cls=resolve_backend))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(wan, "ReplicatedLinear", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(wan, "RMSNorm", lambda *args, **kwargs: torch.nn.Identity())
    _cached_get_attn_backend.cache_clear()
    try:
        backend = get_attn_backend(
            128, torch.bfloat16,
            supported_attention_backends=wan.WanTransformer3DModel._supported_attention_backends,
        )
        assert backend is FlashInferBackend
        cross_attention = cross_attention_cls(dim=256, num_heads=2)
        assert cross_attention.attn.backend is AttentionBackendEnum.FLASHINFER
        assert isinstance(cross_attention.attn.attn_impl, FlashInferImpl)
        assert cross_attention.attn.attn_impl.prefill_backend == prefill_backend
        assert len(selections) == 2
    finally:
        _cached_get_attn_backend.cache_clear()
