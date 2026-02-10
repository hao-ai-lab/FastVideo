# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.pipelines.basic.wan.wan_i2v_pipeline import WanImageToVideoPipeline


class LingBotWorldImageToVideoPipeline(WanImageToVideoPipeline):
    pass


EntryClass = LingBotWorldImageToVideoPipeline
