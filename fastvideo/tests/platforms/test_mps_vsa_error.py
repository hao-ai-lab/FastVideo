# SPDX-License-Identifier: Apache-2.0
"""MPS must reject VSA before a model reaches an incompatible SDPA call."""

import pytest
import torch

from fastvideo.platforms import AttentionBackendEnum
from fastvideo.platforms.mps import MpsPlatform


def test_mps_rejects_video_sparse_attention_with_actionable_error() -> None:
    with pytest.raises(NotImplementedError, match="TORCH_SDPA"):
        MpsPlatform.get_attn_backend_cls(AttentionBackendEnum.VIDEO_SPARSE_ATTN, 64, torch.float16)


def test_mps_resolves_sdpa_for_supported_backend() -> None:
    assert MpsPlatform.get_attn_backend_cls(AttentionBackendEnum.TORCH_SDPA, 64, torch.float16).endswith(
        "SDPABackend"
    )
