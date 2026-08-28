# SPDX-License-Identifier: Apache-2.0
#
# Adapted from:
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/globals.py
#
# FastVideo keeps only yunchang's optional-kernel capability detection here,
# trimmed to the single backend the vendored Ring Attention kernels dispatch
# to (plain FlashAttention). Sequence-parallel process groups are managed by
# fastvideo.distributed.parallel_state.

try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward  # noqa: F401

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
