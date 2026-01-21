# SPDX-License-Identifier: Apache-2.0
"""
HyWorld video diffusion pipeline implementation.

This module contains an implementation of the HyWorld video diffusion pipeline
using the modular pipeline architecture with HyWorld-specific denoising stage
for chunk-based video generation with context frame selection.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        HyWorldDenoisingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage,
                                        HyWorldImageEncodingStage)

logger = init_logger(__name__)


class HyWorldPipeline(ComposedPipelineBase):
    """
    HyWorld video diffusion pipeline.

    This pipeline implements chunk-based video generation with context frame
    selection for 3D-aware generation using HyWorldDenoisingStage.

    Note: HyWorld only uses a single LLM-based text encoder, unlike SDXL-style
    dual encoder setups. The text_encoder_2/tokenizer_2 are not used.
    """

    # Include image_encoder and feature_extractor for I2V support with SigLIP
    # Note: guider (ClassifierFreeGuidance) is not needed - FastVideo handles CFG differently
    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler", "text_encoder_2", "tokenizer_2",
        "image_encoder", "feature_extractor"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with HyWorld-specific denoising stage."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage_primary",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder"), self.get_module("text_encoder_2")],
                           tokenizers=[self.get_module("tokenizer"), self.get_module("tokenizer_2")]
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="image_encoding_stage",
                       stage=HyWorldImageEncodingStage(
                           image_encoder=self.get_module("image_encoder"),
                           image_processor=self.get_module("feature_extractor"),
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=HyWorldDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler"),
                           pipeline=self))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = HyWorldPipeline
