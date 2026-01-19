# SPDX-License-Identifier: Apache-2.0
"""Cosmos 2.5 pipeline entry (staged pipeline)."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (ConditioningStage,
                                        Cosmos25AutoDenoisingStage,
                                        Cosmos25AutoLatentPreparationStage,
                                        DecodingStage, InputValidationStage,
                                        Cosmos25TextEncodingStage,
                                        Cosmos25TimestepPreparationStage)

logger = init_logger(__name__)


class Cosmos2_5Pipeline(ComposedPipelineBase):
    """Cosmos 2.5 video generation pipeline."""

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
        "safety_checker"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        logger.info("Creating Cosmos 2.5 pipeline stages...")

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=Cosmos25TextEncodingStage(
                text_encoder=self.get_module("text_encoder"), ),
        )

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=Cosmos25TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=Cosmos25AutoLatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer"),
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=Cosmos25AutoDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))
        logger.info("Cosmos 2.5 pipeline stages created")


# Entry point for pipeline registry
EntryClass = Cosmos2_5Pipeline
