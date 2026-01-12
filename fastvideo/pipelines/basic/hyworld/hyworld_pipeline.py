# SPDX-License-Identifier: Apache-2.0
"""
HyWorld video diffusion pipeline implementation.

This module contains an implementation of the HyWorld video diffusion pipeline
that inherits from HunyuanVideo15Pipeline, using the HyWorld-specific denoising stage
for chunk-based video generation with context frame selection.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.hunyuan15.hunyuan15_pipeline import (
    HunyuanVideo15Pipeline)
from fastvideo.pipelines.stages import HyWorldDenoisingStage

logger = init_logger(__name__)


class HyWorldPipeline(HunyuanVideo15Pipeline):
    """
    HyWorld video diffusion pipeline.

    This pipeline extends HunyuanVideo15Pipeline by using HyWorldDenoisingStage
    instead of the standard DenoisingStage. The HyWorld denoising stage implements
    chunk-based video generation with context frame selection for 3D-aware generation.

    All other stages remain the same as HunyuanVideo15Pipeline.
    """

    # Include image_encoder and feature_extractor for I2V support with SigLIP
    # Note: guider (ClassifierFreeGuidance) is not needed - FastVideo handles CFG differently
    _required_config_modules = [
        "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae",
        "transformer", "scheduler", "image_encoder", "feature_extractor"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with HyWorld-specific denoising stage."""
        # Call parent to set up all stages
        super().create_pipeline_stages(fastvideo_args)
        
        # Replace the denoising_stage with HyWorldDenoisingStage
        # Find the index of the old denoising_stage in _stages list
        old_denoising_stage = self.denoising_stage
        for i, stage in enumerate(self._stages):
            if stage is old_denoising_stage:
                # Create new HyWorld denoising stage
                new_denoising_stage = HyWorldDenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    pipeline=self)
                
                # Replace in stages list
                self._stages[i] = new_denoising_stage
                # Replace in stage name mapping
                self._stage_name_mapping["denoising_stage"] = new_denoising_stage
                # Replace attribute
                setattr(self, "denoising_stage", new_denoising_stage)
                break


EntryClass = HyWorldPipeline
