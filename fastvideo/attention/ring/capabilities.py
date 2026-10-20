# SPDX-License-Identifier: Apache-2.0
"""Load the optional FA2 private interface only when Ring is enabled."""
from functools import lru_cache


@lru_cache(maxsize=1)
def load_flash_attn():
    try:
        import flash_attn
        from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward
    except (ImportError, OSError) as exc:
        raise RuntimeError("Ring Attention requires FlashAttention-2 (flash_attn.flash_attn_interface). "
                           "Install a compatible FlashAttention-2 package or set ring_size=1.") from exc
    return flash_attn, _flash_attn_forward, _flash_attn_backward
