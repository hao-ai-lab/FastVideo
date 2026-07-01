# SPDX-License-Identifier: Apache-2.0

import sys
import types

import torch

from fastvideo.platforms.cuda import CudaPlatformBase


def test_sm120_auto_attention_uses_flash_backend_with_fa4_default_off(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_FA4", raising=False)

    def has_device_capability(cls, capability, device_id=0):
        del cls, device_id
        return capability in (80, (12, 0))

    flash_attn_module = types.ModuleType("flash_attn")
    flash_backend_module = types.ModuleType("fastvideo.attention.backends.flash_attn")

    class FlashAttentionBackend:
        @staticmethod
        def get_supported_head_sizes():
            return [128]

    flash_backend_module.FlashAttentionBackend = FlashAttentionBackend

    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn_module)
    monkeypatch.setitem(
        sys.modules,
        "fastvideo.attention.backends.flash_attn",
        flash_backend_module,
    )
    monkeypatch.setattr(
        CudaPlatformBase,
        "has_device_capability",
        classmethod(has_device_capability),
    )

    backend = CudaPlatformBase.get_attn_backend_cls(
        selected_backend=None,
        head_size=128,
        dtype=torch.bfloat16,
    )

    assert backend == "fastvideo.attention.backends.flash_attn.FlashAttentionBackend"
