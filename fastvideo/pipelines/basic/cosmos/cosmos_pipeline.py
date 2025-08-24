# SPDX-License-Identifier: Apache-2.0
"""
Cosmos video diffusion pipeline implementation.

This module contains an implementation of the Cosmos video diffusion pipeline
using the modular pipeline architecture.
"""

import os
import numpy as np
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                           CosmosDenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           TextEncodingStage,
                                           TimestepPreparationStage)
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)

logger = init_logger(__name__)


class Cosmos2VideoToWorldPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler", "safety_checker"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):

        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
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
                       stage=CosmosDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))
        


EntryClass = Cosmos2VideoToWorldPipeline