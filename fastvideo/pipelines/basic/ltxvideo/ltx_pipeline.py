# SPDX-License-Identifier: Apache-2.0
"""
LTX video diffusion pipeline implementation.

This module contains an implementation of the LTX video diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        DenoisingStage, InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)


class LTXPipeline(ComposedPipelineBase):
    """
    LTX video diffusion pipeline
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize pipeline-specific components."""
        pass

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        # Add ImageVAEEncodingStage for I2V (conditional based on input)
        # Before LatentPreparation for I2V
        # if fastvideo_args.pipeline_config.ltx_i2v_mode:
        #     self.add_stage(
        #         stage_name="image_vae_encoding_stage",
        #         stage=LTXImageVAEEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        # Only add LatentPreparation for T2V (I2V creates latents in ImageVAEEncoding)
        if not fastvideo_args.pipeline_config.ltx_i2v_mode:
            self.add_stage(stage_name="latent_preparation_stage",
                           stage=LatentPreparationStage(
                               scheduler=self.get_module("scheduler"),
                               transformer=self.get_module("transformer",
                                                           None)))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler"),
                           pipeline=self))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae"),
                                           pipeline=self))


EntryClass = LTXPipeline
