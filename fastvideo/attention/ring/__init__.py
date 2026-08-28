# SPDX-License-Identifier: Apache-2.0
#
# This package vendors Ring Attention kernels for FastVideo's Ring Attention
# integration. See each module's header for its upstream source and license.
#
# Only ``ring_flash_attn`` (Ring Attention, optionally combined with Ulysses
# as the USP hybrid, over plain FlashAttention) is vendored here — it is what
# ``fastvideo.attention.layer`` uses. Upstream yunchang also has zig-zag,
# striped, variable-length, PyTorch, NPU, and FlashInfer variants; FastVideo
# does not vendor them (they would depend on the ``yunchang`` package or
# other optional backends FastVideo does not want as hard runtime
# dependencies), and already has its own backend-dispatch system
# (``fastvideo.attention.selector``) for backends other than FlashAttention.

from .ring_flash_attn import ring_flash_attn_func

__all__ = [
    "ring_flash_attn_func",
]
