# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the family-local causal Wan transformer."""

from fastvideo.models.wan.causal_transformer import (
    GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES,
    CausalWanSelfAttention,
    CausalWanTransformer3DModel,
    CausalWanTransformerBlock,
    EntryClass,
)

__all__ = [
    "GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES",
    "CausalWanSelfAttention",
    "CausalWanTransformer3DModel",
    "CausalWanTransformerBlock",
    "EntryClass",
]
