# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.v1.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.v1.pipelines.stages import (ConditioningStage, DecodingStage,
                                           DenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           TextEncodingStage,
                                           TimestepPreparationStage)

logger = init_logger(__name__)


class WanPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Wan video diffusion pipeline with LoRA support.
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # We use UniPCMScheduler from Wan2.1 official repo, not the one in diffusers.
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


class WanValidationPipeline(ComposedPipelineBase):
    """
    Validation pipeline for Wan2.1, assumes that the input are preprocess latents.
    """
    _required_config_modules = ["vae", "scheduler", "transformer"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanPipeline
