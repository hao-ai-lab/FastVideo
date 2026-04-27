# SPDX-License-Identifier: Apache-2.0
"""
Cosmos video diffusion pipeline implementation.

This module contains an implementation of the Cosmos video diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, CosmosDenoisingStage, CosmosLatentPreparationStage,
                                        DecodingStage, InputValidationStage, TextEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)


class Cosmos2VideoToWorldPipeline(ComposedPipelineBase):

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler", "safety_checker"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift,
            use_karras_sigmas=True,
        )
        scheduler.config.sigma_max = 80.0
        scheduler.config.sigma_min = 0.002
        scheduler.config.sigma_data = 1.0
        scheduler.config.final_sigmas_type = "sigma_min"
        scheduler.sigma_max = 80.0
        scheduler.sigma_min = 0.002
        scheduler.sigma_data = 1.0
        self.modules["scheduler"] = scheduler

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=CosmosLatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                          transformer=self.get_module("transformer"),
                                                          vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=CosmosDenoisingStage(transformer=self.get_module("transformer"),
                                                  scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = Cosmos2VideoToWorldPipeline
