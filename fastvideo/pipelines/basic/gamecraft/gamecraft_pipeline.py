# SPDX-License-Identifier: Apache-2.0
"""
HunyuanGameCraft video diffusion pipeline implementation.

This module implements the HunyuanGameCraft pipeline for camera/action-conditioned
video generation with the modular pipeline architecture.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from fastvideo.pipelines.stages.gamecraft_denoising import GameCraftDenoisingStage

logger = init_logger(__name__)


class HunyuanGameCraftPipeline(ComposedPipelineBase):
    """
    Pipeline for HunyuanGameCraft video generation.
    
    This pipeline supports:
    - Text-to-video generation with camera/action conditioning
    - Autoregressive generation with history frames
    - 33-channel input (16 latent + 16 gt_latent + 1 mask)
    - CameraNet for encoding Plücker coordinates
    """

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
        )

        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage(),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=GameCraftDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )


EntryClass = HunyuanGameCraftPipeline
