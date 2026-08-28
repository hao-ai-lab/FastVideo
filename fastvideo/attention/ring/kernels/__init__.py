# SPDX-License-Identifier: Apache-2.0
#
# Adapted from:
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/kernels/__init__.py
#
# FastVideo keeps only the FA dispatch path used by the vendored pure-Ring
# Attention implementation (fastvideo.attention.layer). Upstream yunchang
# dispatches to many more backends (FA3, FlashInfer, aiter, SageAttention,
# torch SDPA); FastVideo already has its own backend-dispatch system
# (fastvideo.attention.selector / backends/), so those paths were dropped
# here rather than carried as a second, unreachable dispatcher.

from __future__ import annotations

from enum import Enum

from .attention import flash_attn_backward, flash_attn_forward


class AttnType(Enum):
    FA = "fa"


def select_flash_attn_impl(impl_type: AttnType, stage: str = "fwd-bwd"):
    if impl_type != AttnType.FA:
        raise ValueError(f"Unknown flash attention implementation: {impl_type}")
    if stage == "fwd-only":
        return flash_attn_forward
    elif stage == "bwd-only":
        return flash_attn_backward
    else:
        raise ValueError(f"Unknown stage: {stage}")


__all__ = [
    "AttnType",
    "select_flash_attn_impl",
]
