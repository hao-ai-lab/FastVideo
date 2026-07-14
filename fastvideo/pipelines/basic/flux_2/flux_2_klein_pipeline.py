# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: https://github.com/sglang-ai/sglang
"""
Flux2 Klein image generation pipeline (distilled, 4-step, no guidance).
"""

from fastvideo.configs.pipelines.flux_2 import Flux2KleinPipelineConfig
from fastvideo.pipelines.basic.flux_2.flux_2_pipeline import Flux2Pipeline


class Flux2KleinPipeline(Flux2Pipeline):
    """Flux2 Klein image diffusion pipeline (distilled, 4-step, no guidance)."""

    pipeline_config_cls: type[Flux2KleinPipelineConfig] = Flux2KleinPipelineConfig


EntryClass = Flux2KleinPipeline
