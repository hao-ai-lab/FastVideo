# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.platforms.cuda import CudaPlatformBase


def test_sm120_auto_attention_avoids_flash_attn_cute(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ALLOW_FA4_SM120", raising=False)

    def has_device_capability(cls, capability, device_id=0):
        del cls, device_id
        return capability in (80, (12, 0))

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

    assert backend == "fastvideo.attention.backends.sdpa.SDPABackend"
