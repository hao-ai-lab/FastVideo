# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for Wan's family-owned causal sampling stages."""

from fastvideo.pipelines.basic.wan.stages.causal_denoising import (
    CausalDMDDenosingStage,
    CausalDenoisingStage,
    _get_transformer_attr,
)

__all__ = ["CausalDMDDenosingStage", "CausalDenoisingStage", "_get_transformer_attr"]
