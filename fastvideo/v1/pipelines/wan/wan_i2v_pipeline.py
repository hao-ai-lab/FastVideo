# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase

# isort: off
from fastvideo.v1.pipelines.stages import (
    CLIPImageEncodingStage, ConditioningStage, DecodingStage, DenoisingStage,
    EncodingStage, InputValidationStage, LatentPreparationStage,
    T5EncodingStage, TimestepPreparationStage)
# isort: on

logger = init_logger(__name__)


class WanImageToVideoPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler", \
        "image_encoder", "image_processor"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=T5EncodingStage(
                           text_encoder=self.get_module("text_encoder"),
                           tokenizer=self.get_module("tokenizer"),
                       ))

        self.add_stage(stage_name="image_encoding_stage",
                       stage=CLIPImageEncodingStage(
                           image_encoder=self.get_module("image_encoder"),
                           image_processor=self.get_module("image_processor"),
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=EncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """
        Initialize the pipeline.
        """
        vae_scale_factor = self.get_module("vae").spatial_compression_ratio
        fastvideo_args.vae_scale_factor = vae_scale_factor

        num_channels_latents = self.get_module("transformer").out_channels
        fastvideo_args.num_channels_latents = num_channels_latents


EntryClass = WanImageToVideoPipeline
