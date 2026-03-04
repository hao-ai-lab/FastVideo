# SPDX-License-Identifier: Apache-2.0
"""
Legacy Wan DMD pipeline entrypoint.

Historically FastVideo exposed a dedicated `WanDMDPipeline` class that wired a
stochastic (SDE-style) denoising loop. Phase 3.2 makes sampling loop selection
explicit via `pipeline_config.sampler_kind`, so this file becomes a thin
compatibility wrapper around `WanPipeline`.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline


class WanDMDPipeline(WanPipeline):
    """Compatibility wrapper for SDE sampling on Wan."""

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        fastvideo_args.pipeline_config.sampler_kind = "sde"
        return super().initialize_pipeline(fastvideo_args)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        fastvideo_args.pipeline_config.sampler_kind = "sde"
        return super().create_pipeline_stages(fastvideo_args)


EntryClass = WanDMDPipeline

