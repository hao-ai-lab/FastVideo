# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan video diffusion pipeline implementation.

This module contains an implementation of the Hunyuan video diffusion pipeline
using the modular pipeline architecture.
"""

from diffusers.image_processor import VaeImageProcessor

from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger

from ..composed_pipeline_base import ComposedPipelineBase
from ..stages import (CLIPTextEncodingStage, ConditioningStage, DecodingStage,
                      DenoisingStage, InputValidationStage,
                      LatentPreparationStage, LlamaEncodingStage,
                      TimestepPreparationStage)

# TODO(will): move PRECISION_TO_TYPE to better place

logger = init_logger(__name__)


class HunyuanVideoPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae",
        "transformer", "scheduler"
    ]

    def create_pipeline_stages(self, inference_args: InferenceArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage_primary",
                       stage=LlamaEncodingStage(
                           text_encoder=self.get_module("text_encoder"),
                           tokenizer=self.get_module("tokenizer"),
                       ))

        self.add_stage(stage_name="prompt_encoding_stage_secondary",
                       stage=CLIPTextEncodingStage(
                           text_encoder=self.get_module("text_encoder_2"),
                           tokenizer=self.get_module("tokenizer_2"),
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

    def initialize_pipeline(self, inference_args: InferenceArgs):
        """
        Initialize the pipeline.
        """
        vae_scale_factor = 2**(len(self.get_module("vae").block_out_channels) -
                               1)
        inference_args.vae_scale_factor = vae_scale_factor

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor)
        self.add_module("image_processor", self.image_processor)

        num_channels_latents = self.get_module("transformer").in_channels
        inference_args.num_channels_latents = num_channels_latents


EntryClass = HunyuanVideoPipeline
