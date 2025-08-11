# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

# isort: off
from fastvideo.pipelines.stages import (
    ImageEncodingStage, ConditioningStage, DecodingStage, DenoisingStage,
    ImageVAEEncodingStage, InputValidationStage, LatentPreparationStage,
    TextEncodingStage, TimestepPreparationStage)
# isort: on
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)

logger = init_logger(__name__)


class WanImageToVideoPipeline(LoRAPipeline, ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler", \
        "image_encoder", "image_processor"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        if (self.get_module("image_encoder") is not None
                and self.get_module("image_processor") is not None):
            self.add_stage(
                stage_name="image_encoding_stage",
                stage=ImageEncodingStage(
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
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=ImageVAEEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanImageToVideoPipeline
