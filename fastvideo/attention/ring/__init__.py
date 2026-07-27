# SPDX-License-Identifier: Apache-2.0
#
# This package vendors Ring Attention kernels for FastVideo's Ring Attention
# integration. See each module's header for its upstream source and license.
#
# Only ``ring_flash_attn`` (pure Ring Attention over plain FlashAttention) is
# wired up and exported here — it is what the initial pure-Ring integration
# in ``fastvideo.attention.layer`` uses. The other vendored variants
# (variable-length, zig-zag, striped, PyTorch/NPU/FlashInfer backends) are
# kept in this directory for future work but are intentionally not imported
# here yet: several of them depend on the ``yunchang`` package or other
# optional backends (FlashInfer, torch_npu) that FastVideo does not want as
# hard runtime dependencies. Import them directly from their modules if you
# need them and have their dependencies installed.

from .ring_flash_attn import (
    ring_flash_attn_func,
    ring_flash_attn_kvpacked_func,
    ring_flash_attn_qkvpacked_func,
)

__all__ = [
    "ring_flash_attn_func",
    "ring_flash_attn_kvpacked_func",
    "ring_flash_attn_qkvpacked_func",
]
