# SPDX-License-Identifier: Apache-2.0
import torch

from fastvideo.attention import selector
from fastvideo.platforms import AttentionBackendEnum


def test_get_attn_backend_cache_respects_forced_backend(monkeypatch):
    selector._cached_get_attn_backend.cache_clear()

    class _FakePlatform:

        @staticmethod
        def get_attn_backend_cls(selected_backend, head_size, dtype):
            if selected_backend == AttentionBackendEnum.ATTN_QAT_TRAIN:
                return "builtins.int"
            return "builtins.str"

    monkeypatch.setattr("fastvideo.platforms.current_platform", _FakePlatform())

    supported_backends = (AttentionBackendEnum.FLASH_ATTN,
                          AttentionBackendEnum.TORCH_SDPA)

    default_impl = selector.get_attn_backend(128, torch.bfloat16, supported_backends)
    with selector.global_force_attn_backend_context_manager(
            AttentionBackendEnum.ATTN_QAT_TRAIN):
        forced_impl = selector.get_attn_backend(128, torch.bfloat16,
                                                supported_backends)
    default_impl_after = selector.get_attn_backend(128, torch.bfloat16,
                                                   supported_backends)

    assert default_impl is str
    assert forced_impl is int
    assert default_impl_after is str
