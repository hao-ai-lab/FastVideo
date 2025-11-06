# SPDX-License-Identifier: Apache-2.0
"""
LongCat video diffusion pipeline implementation (Phase 1: Wrapper).

This module contains a wrapper implementation of the LongCat video diffusion pipeline
using FastVideo's modular pipeline architecture with the original LongCat modules.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)

logger = init_logger(__name__)


class LongCatPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    LongCat video diffusion pipeline with LoRA support.
    
    Phase 1 implementation using wrapper modules from third_party/longcat_video.
    This validates the pipeline infrastructure before full FastVideo integration.
    """

    _required_config_modules = [
        "text_encoder",
        "tokenizer", 
        "vae",
        "transformer",
        "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize LongCat-specific components."""
        # LongCat uses FlowMatchEulerDiscreteScheduler which is already loaded
        # from the model_index.json, so no need to override
        pass

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")
            )
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None)
            )
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2", None),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self
            )
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae"),
                pipeline=self
            )
        )


EntryClass = LongCatPipeline


