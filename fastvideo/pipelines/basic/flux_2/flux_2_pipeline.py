# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: https://github.com/sglang-ai/sglang
"""
Flux2 image generation pipeline implementation.

This module contains an implementation of the Flux2 image diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.basic.flux_2.flux_2_timestep_preparation import (
    Flux2TimestepPreparationStage,
)
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
)

logger = init_logger(__name__)


class Flux2Pipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Flux2 image diffusion pipeline with LoRA support.
    """

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(
            stage_name="image_encoding_stage",
            stage=ImageVAEEncodingStage(
                vae=self.get_module("vae"),
            ),
        )

        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage(),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=Flux2TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self,
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            ),
        )


EntryClass = Flux2Pipeline
