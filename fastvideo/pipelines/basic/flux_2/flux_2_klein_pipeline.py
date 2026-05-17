# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: https://github.com/sglang-ai/sglang
"""
Flux2 Klein image generation pipeline (distilled, 4-step, no guidance).
"""

from fastvideo.pipelines.basic.flux_2.flux_2_pipeline import Flux2Pipeline


class Flux2KleinPipeline(Flux2Pipeline):
    """Flux2 Klein image diffusion pipeline (distilled, 4-step, no guidance)."""


EntryClass = Flux2KleinPipeline
