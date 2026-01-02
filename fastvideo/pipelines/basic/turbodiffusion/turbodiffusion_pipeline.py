# SPDX-License-Identifier: Apache-2.0
"""
TurboDiffusion Video Pipeline Implementation.

This module contains an implementation of the TurboDiffusion video diffusion pipeline
for 1-4 step video generation using rCM (recurrent Consistency Model) sampling
with SLA (Sparse-Linear Attention).
"""

import os
import re
import torch

from fastvideo.utils import maybe_download_model

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_rcm import RCMScheduler
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        DenoisingStage, InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)


class TurboDiffusionPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    TurboDiffusion video pipeline for 1-4 step generation.
    
    Uses RCM scheduler and SLA attention for fast, high-quality video generation.
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # Use RCM scheduler for TurboDiffusion
        logger.info("Initializing RCM scheduler for TurboDiffusion")
        self.modules["scheduler"] = RCMScheduler(sigma_max=80.0)
        
        # Store checkpoint path for later loading
        self._turbodiffusion_checkpoint = getattr(
            fastvideo_args, 'turbodiffusion_checkpoint', None
        )


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
                           transformer_2=self.get_module("transformer_2", None),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae"),
                           pipeline=self))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae"),
                                           pipeline=self))


EntryClass = TurboDiffusionPipeline
