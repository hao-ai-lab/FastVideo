# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the canonical :mod:`fastvideo.models.wan.transformer`."""

from fastvideo.models.wan.transformer import (
    EntryClass,
    LayerNormScaleShift,
    PatchEmbed,
    WanI2VCrossAttention,
    WanImageEmbedding,
    WanSelfAttention,
    WanT2VCrossAttention,
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
    WanTransformerBlock,
    WanTransformerBlock_VSA,
)

__all__ = [
    "EntryClass",
    "LayerNormScaleShift",
    "PatchEmbed",
    "WanI2VCrossAttention",
    "WanImageEmbedding",
    "WanSelfAttention",
    "WanT2VCrossAttention",
    "WanTimeTextImageEmbedding",
    "WanTransformer3DModel",
    "WanTransformerBlock",
    "WanTransformerBlock_VSA",
]
