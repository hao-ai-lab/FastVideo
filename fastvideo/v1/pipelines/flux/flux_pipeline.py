# SPDX-License-Identifier: Apache-2.0
"""
Flux image diffusion pipeline implementation.

This module contains an implementation of the Flux image diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.flux.custom_stages import (
    DenoisingPostprocessingStage, DenoisingPreprocessingStage,
    TimestepsPreparationPreStage)
from fastvideo.v1.pipelines.stages import (ConditioningStage, DecodingStage,
                                           InputValidationStage,
                                           LatentPreparationStage,
                                           TextEncodingStage,
                                           TimestepPreparationStage,
                                           DenoisingStage)

logger = init_logger(__name__)


class FluxPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae",
        "transformer", "scheduler"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[
                               self.get_module("text_encoder"),
                               self.get_module("text_encoder_2"),
                           ],
                           tokenizers=[
                               self.get_module("tokenizer"),
                               self.get_module("tokenizer_2"),
                           ],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timesteps_preparation_pre_stage",
                       stage=TimestepsPreparationPreStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="modulation",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="denoising_preprocessing_stage",
                       stage=DenoisingPreprocessingStage())

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="denoising_postprocessing_stage",
                       stage=DenoisingPostprocessingStage())

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = FluxPipeline
