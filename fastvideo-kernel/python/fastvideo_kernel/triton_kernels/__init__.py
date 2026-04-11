"""Triton kernel entrypoints exposed by ``fastvideo_kernel``."""

from .fused_attention import attention as fused_attention

__all__ = ["fused_attention"]
